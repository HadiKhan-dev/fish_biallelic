"""The read state: which homologues a sample's reads come from, carried along the contig by read survival, for
libraries whose reads span many sites (the painting's read state, PaintingParameters.read_classes > 1).

A sequenced molecule covers every site it spans and comes from ONE homologue. With long reads a true heterozygote is
often covered by one homologue's reads for tens of kb, and the sites' genotype likelihoods, multiplied as if
independent, charge that one chance event once per site. This module models the reads' origin instead, from the
per-site REF/ALT counts and the library's measured read sharing alone (no read identities).

Per sample and site of the evidence grid (every called site):
- covering depth N = n + o: the called depth n plus a loss offset o in {0, ..., offsets - 1} (bases of covering reads
  lost to deletions or third-base calls), P(n | N) = Bin(N - n; N, delta);
- the covering split (K1, K2 = N - K1) between the two homologues, tracked by CLASS (split_class_map): the exact
  count of the minor homologue up to ``minor``, the rest 'balanced' (minor = 0: {balanced, homologue 1 only,
  homologue 2 only}); the number of classes 2 * minor + 3 does not depend on depth;
- between consecutive sites d bp apart each covering read survives with probability phi(d) (the library's fitted
  read-length law, fit_survival) and new reads arrive as Poisson(m (1 - phi(d))) per homologue (m: the sample's
  covering depth per homologue); the transitions are conditioned on the covering depths at both sites.
The loss offsets do not depend on the founders, so their posterior comes from the depths alone (offset_posteriors, a
depth-only HMM over the offsets); the class chain then uses per-gap operators averaged over the joint law of the offsets
and the source class (class_operators) and per-site allele likelihoods averaged over the class-conditional offset
posterior (class_likelihoods). Consumers that read a subset of the sites compose the operators over the skipped sites
(compose_operators): the chain is defined on the evidence grid, never by looking the survival up at a coarser grid's
gaps. Covering depths at and above the deep bound (deep_depth) are 'balanced' with certainty to
ReadStateParameters.deep_tail: there the class likelihoods take closed forms, and a sample whose every site of a span
is deep bypasses the read state exactly (deep_rows), so neither tables nor per-site work grow with depth.

With no read sharing (phi = 0 at every gap) the class operators' rows are the class law of the next site, and a
chain fed the GL-anchored read factor (painting.read_factor) reduces exactly to independent sites.

The NumPy functions (offset_posteriors, class_operators, class_likelihoods) are the reference forms; the *_fast
functions compute the same values through Numba kernels over samples and independent site blocks (class_likelihoods_fast tabulates the
split sums once per distinct (called depth, ALT count); class_likelihoods_sitewise is its per-site check).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from typing import TYPE_CHECKING

import numpy as np
from numba import prange, types
from scipy.optimize import nnls
from scipy.stats import binom, hypergeom, norm, poisson

from ...core.concurrency import live_threads
from ...core.kernels import bounded_team, kernel

if TYPE_CHECKING:
    from ..painting.parameters import ReadStateParameters

__all__ = [
    "ReadSurvival",
    "ReadTables",
    "length_grid",
    "no_survival",
    "fit_survival",
    "split_class_map",
    "split_weights",
    "depth_table",
    "class_table",
    "deep_depth",
    "deep_rows",
    "table_depth",
    "build_tables",
    "offset_posteriors",
    "offset_posteriors_fast",
    "offset_pilot_width",
    "adaptive_offset_operators",
    "class_operators",
    "class_operators_fast",
    "site_laws",
    "compose_operators",
    "class_likelihoods",
    "class_likelihoods_fast",
    "class_likelihoods_sitewise",
    "library_groups",
]


@dataclass(frozen=True)
class ReadSurvival:
    """A library's read-length law fitted to its read-sharing curve: phi(d) = E[(L - d)+] / E[L] (the share of a
    site's covering reads that still cover a site d bp away) and the intermittent loss delta (the share of covering
    reads without a called base at a site). ``weights`` >= 0 on ``grid`` (bp): sum_j w_j (l_j - d)+ = (1 - delta)
    phi(d). status: 'fitted' or 'none' (no measurable sharing: independent sites)."""

    grid: np.ndarray
    weights: np.ndarray
    loss: float                 # intermittent loss used by the model, in [0, maximum_loss]
    status: str
    fitted_loss: float = 0.0    # 1 - the fitted curve at d = 0, unclipped (diagnostic)

    def survival(self, gap) -> np.ndarray:
        gap = np.asarray(gap, dtype=np.float64)
        level = float(self.weights @ self.grid)
        if self.status != "fitted" or level <= 0.0:
            return np.zeros(gap.shape)
        return np.clip(np.maximum(self.grid - gap[..., None], 0.0) @ self.weights / level, 0.0, 1.0)


def length_grid(parameters: ReadStateParameters) -> np.ndarray:
    """float64[J] the survival fit's log-spaced read-length grid (bp)."""
    return np.geomspace(parameters.length_grid_low_bp, parameters.length_grid_high_bp, parameters.length_grid_points)


def no_survival(parameters: ReadStateParameters) -> ReadSurvival:
    """The law of a library without measurable read sharing: phi = 0 at every gap, no loss (independent sites)."""
    grid = length_grid(parameters)
    return ReadSurvival(grid, np.zeros(len(grid)), 0.0, "none")


def _bin_average_overlap(grid, low, high):
    """[K, J] average over d in [low_k, high_k) of (l_j - d)+."""
    length = grid[None, :]
    a, b = low[:, None], high[:, None]
    return np.where(
        length >= b, length - (a + b) / 2.0,
        np.where(length > a, (length - a) ** 2 / (2.0 * np.maximum(b - a, 1e-9)), 0.0),
    )


def usable_bins(sharing) -> int:
    """The lag bins of a ReadSharing record with a finite estimate and a positive standard error (the bins a
    detection test reads)."""
    excess = np.asarray(sharing.excess, dtype=np.float64)
    error = np.asarray(sharing.standard_error, dtype=np.float64)
    return int(np.count_nonzero(np.isfinite(excess) & np.isfinite(error) & (error > 0)))


def structural_bins(sharing) -> int:
    """The lag bins of a contig that hold site pairs (ReadSharing.pairs > 0): every bin any estimate on the contig's
    sites can measure, whatever samples it uses (the pairs depend on the positions only), so a fixed count for the
    all-sample fit and every candidate library split alike, bins unusable in the all-sample estimate included."""
    return int(np.count_nonzero(np.asarray(sharing.pairs) > 0))


def bin_summary(sharing) -> dict:
    """A ReadSharing record's detection statistic: dict(max_z: the largest usable lag bin's excess / SE (nan without
    usable bins), bin_bp: that bin's lag interval [low, high) in bp (None without usable bins), usable: usable_bins)."""
    excess = np.asarray(sharing.excess, dtype=np.float64)
    error = np.asarray(sharing.standard_error, dtype=np.float64)
    edges = np.asarray(sharing.edges, dtype=np.int64)
    usable = np.isfinite(excess) & np.isfinite(error) & (error > 0)
    if not usable.any():
        return dict(max_z=float("nan"), bin_bp=None, usable=0)
    z = np.where(usable, excess / np.where(usable, error, 1.0), -np.inf)
    k = int(np.argmax(z))
    return dict(max_z=float(z[k]), bin_bp=(int(edges[k]), int(edges[k + 1])), usable=int(usable.sum()))


def strongest_bin(sharing) -> float:
    """The largest usable lag bin's excess / standard error of a ReadSharing record (nan without usable bins): the
    detection statistic a fit compares with z* (bin_summary's max_z)."""
    return bin_summary(sharing)["max_z"]


def near_detection_edge(strength: float, z: float) -> bool:
    """Whether a statistic (a library's largest excess / SE, or a candidate split's contrast) lies within a factor
    of two of the detection threshold z* on either side (z* / 2 <= strength <= 2 z*): the decision it makes is
    close to the detection edge."""
    return bool(np.isfinite(strength) and 0.5 * float(z) <= float(strength) <= 2.0 * float(z))


def candidate_splits(samples: int, parameters: ReadStateParameters) -> int:
    """The most candidate library splits library_groups can test on one contig: 2 ** library_levels - 1 (a full
    divisive tree) when the run has the 2 * library_minimum_samples samples a split needs, else none."""
    if int(samples) < 2 * int(parameters.library_minimum_samples):
        return 0
    return 2 ** int(parameters.library_levels) - 1


def family_tests(bins, splits: int) -> int:
    """M, the run's read-sharing tests: every one-sided lag-bin test the detection can make. Per analysed contig
    (``bins``: its structural_bins) the all-sample fit tests each bin once, and each of at most ``splits`` candidate
    library splits (candidate_splits) tests each bin four times: the two parts' difference in both directions and
    each part's own fit. A bound fixed before any test is read (never only the tests that survived a screen), so z*
    does not depend on which splits are accepted."""
    return int(sum(int(b) for b in bins)) * (1 + 4 * int(splits))


def family_z(alpha: float, tests: int) -> float:
    """z* = Phi^-1(1 - alpha / M): the one-sided per-test threshold of a family-wise error ``alpha`` over ``tests``
    tests (Bonferroni)."""
    return float(norm.isf(float(alpha) / max(int(tests), 1)))


def fit_survival(sharing, parameters: ReadStateParameters, z: float) -> ReadSurvival:
    """Fit the read-length law to a ReadSharing record (science.evidence.read_sharing): every measured lag bin's signed
    estimate (``excess``) weighted by its delete-one-sample jackknife standard error, each bin's model value the
    average of (l - d)+ over its lags, non-negative least squares on the length grid. Bins beyond the published reach
    are measured too and enter as measured (the reach is a reporting limit, not zero overlap). The loss is 1 - the
    fitted curve at d = 0, clipped to [0, maximum_loss]; the unclipped value is kept as ``fitted_loss``. Sharing is
    detected when some usable bin's excess exceeds ``z`` standard errors (the run's family-wise z*, family_z); else the
    library shows none (no_survival)."""
    edges = np.asarray(sharing.edges, dtype=np.float64)
    excess = np.asarray(sharing.excess, dtype=np.float64)
    error = np.asarray(sharing.standard_error, dtype=np.float64)
    usable = np.isfinite(excess) & np.isfinite(error) & (error > 0)
    none = no_survival(parameters)
    if usable.sum() < parameters.minimum_sharing_bins or not np.any(
        excess[usable] > float(z) * error[usable]
    ):
        return none
    grid = length_grid(parameters)
    design = _bin_average_overlap(grid, edges[:-1][usable], edges[1:][usable])
    weight = 1.0 / error[usable]
    fitted, _ = nnls(design * weight[:, None], excess[usable] * weight)
    level = float(fitted @ grid)
    if level <= 0.0:
        return none
    # the model needs a probability: a fitted level above one (loss below zero, sampling noise on short-read libraries)
    # means no measurable loss; the unclipped value is reported (fitted_loss)
    return ReadSurvival(grid, fitted, float(min(max(1.0 - level, 0.0), parameters.maximum_loss)), "fitted", 1.0 - level)


def split_class_map(nmax: int, minor: int) -> tuple[np.ndarray, int]:
    """cls[N, K1]: class of the covering split (K1, N - K1). 0 = 'balanced': equal splits (empty coverage included) or
    both counts above ``minor``; 1 + j = homologue 2 is the minor one with exactly j <= minor reads (fewer than
    homologue 1's); minor + 2 + j = homologue 1 is the minor one with exactly j reads. One permutation
    (painting.read_factor.swap_classes) exchanges the homologues at every depth; minor = 0 gives {balanced, homologue 1
    only, homologue 2 only}. Returns (cls, number of classes)."""
    n_classes = 2 * minor + 3
    cls = np.full((nmax + 1, nmax + 1), -1, dtype=np.int64)
    for total in range(nmax + 1):
        for first in range(total + 1):
            second = total - first
            if first == second or (first > minor and second > minor):
                cls[total, first] = 0
            elif second < first:
                cls[total, first] = 1 + second
            else:
                cls[total, first] = minor + 2 + first
    return cls, n_classes


def split_weights(nmax: int, cls: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """(within[N, K1], law[N, C]): the split's law Bin(K1; N, 1/2) renormalised within its class, and P(class | N)."""
    counts = np.arange(nmax + 1)
    law = np.where(counts[None, :] <= counts[:, None], binom.pmf(counts[None, :], counts[:, None], 0.5), 0.0)
    within = np.zeros_like(law)
    class_law = np.zeros((nmax + 1, n_classes))
    for total in range(nmax + 1):
        first = np.arange(total + 1)
        member = cls[total, first]
        mass = np.bincount(member, weights=law[total, first], minlength=n_classes)
        within[total, first] = law[total, first] / mass[member]
        class_law[total] = mass
    return within, class_law


@lru_cache(maxsize=256)
def _split_law(nmax: int, minor: int) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    cls, n_classes = split_class_map(nmax, minor)
    within, law = split_weights(nmax, cls, n_classes)
    return cls, n_classes, within, law


def _split_tables(nmax: int, minor: int) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """(cls, n_classes, within, law) of split_class_map and split_weights at (nmax, minor): the same arrays, computed
    once per process (their Python loops hold the GIL that concurrent window and batch threads share) and copied for
    each caller, which owns its arrays."""
    cls, n_classes, within, law = _split_law(int(nmax), int(minor))
    return cls.copy(), n_classes, within.copy(), law.copy()


def survivor_laws(survival: float, nmax: int) -> np.ndarray:
    """float64[nmax + 1, nmax + 1]: row K is the survivors' law Bin(K, survival) of covering depth K (zero beyond K),
    the rows that depth_table and class_table convolve with the arrivals (a function of the survival alone: one set
    serves every arrival depth)."""
    counts = np.arange(nmax + 1)
    return binom.pmf(counts, counts[:, None], survival)


def survivor_rows(survival: float, nmax: int) -> list:
    """[Bin(K, survival) as float64[K + 1] for K = 0..nmax]: survivor_laws' rows within their support (the reference
    forms' input)."""
    return [law[:k + 1] for k, law in enumerate(survivor_laws(survival, nmax))]


def depth_table(survival: float, arrivals: float, nmax: int, rows: list | None = None) -> np.ndarray:
    """P(N' | N) [nmax + 1, nmax + 1] of the covering depth: Bin(N, survival) survivors + Poisson(2 arrivals).
    ``rows``: survivor_rows(survival, nmax), when the caller holds them."""
    counts = np.arange(nmax + 1)
    rows = survivor_rows(survival, nmax) if rows is None else rows
    return _arrival_table(rows, poisson.pmf(counts, 2.0 * arrivals))


def _arrival_table(rows: list, new: np.ndarray) -> np.ndarray:
    """Depth transitions for one or both homologues, given their survivor and Poisson arrival laws (the reference form
    of _arrival_kernel)."""
    table = np.empty((len(rows), len(new)))
    for k, law in enumerate(rows):
        table[k] = np.convolve(law, new)[:len(new)]
    return table


def class_tables(ones: np.ndarray, nmax: int, minor: int) -> np.ndarray:
    """[T, N, N', C, C] the class transitions of class_table for T homologue laws ``ones`` [T, N, N'] at once (one
    NumPy pass per covering depth for all of them: the same products, the same matrix product per law and the same
    ordered sums as one law alone)."""
    cls, n_classes = split_class_map(nmax, minor)
    within, _ = split_weights(nmax, cls, n_classes)
    n_tables = ones.shape[0]
    next_total, next_first = np.nonzero(np.tril(np.ones((nmax + 1, nmax + 1), dtype=bool)))
    next_second = next_total - next_first
    target = next_total * n_classes + cls[next_total, next_first]
    stacked = (np.arange(n_tables)[:, None] * ((nmax + 1) * n_classes) + target[None, :]).ravel()
    table = np.zeros((n_tables, nmax + 1, nmax + 1, n_classes, n_classes))
    for total in range(nmax + 1):
        first = np.arange(total + 1)
        joint = ones[:, first][:, :, next_first] * ones[:, total - first][:, :, next_second]
        lump = np.zeros((n_classes, total + 1))
        lump[cls[total, first], first] = within[total, first]
        from_class = np.matmul(lump, joint)                                 # [T, C, M]: one product per law
        for c in range(n_classes):
            row = np.bincount(stacked, weights=from_class[:, c].ravel(), minlength=n_tables * (nmax + 1) * n_classes)
            row = row.reshape(n_tables, nmax + 1, n_classes)
            mass = row.sum(axis=2, keepdims=True)
            table[:, total, :, c, :] = np.where(mass > 0, row / np.where(mass > 0, mass, 1.0), 1.0 / n_classes)
    return table


def class_table(survival: float, arrivals: float, nmax: int, minor: int, rows: list | None = None) -> np.ndarray:
    """[N, N', C, C] class transitions of the covering split, conditioned on the covering depths at both sites: per
    homologue Bin(K, survival) survivors + Poisson(arrivals) new reads, from each class's within-class split law.
    ``rows``: survivor_rows(survival, nmax), when the caller holds them."""
    rows = survivor_rows(survival, nmax) if rows is None else rows
    one = _arrival_table(rows, poisson.pmf(np.arange(nmax + 1), arrivals))
    return class_tables(one[None], nmax, minor)[0]


def deep_depth(minor: int, deep_tail: float) -> int:
    """The smallest covering depth N at which, under a 50:50 split, either homologue holding <= minor reads has
    probability below ``deep_tail``: from there on the split class is 'balanced' with certainty to that precision (at
    1e-12: minor 0, 1, 2 give 42, 47, 52). Fixed, so neither tables nor per-site work grow with sequencing depth."""
    n = 1
    while 2.0 * sum(math.comb(n, k) for k in range(minor + 1)) * 0.5 ** n >= deep_tail:
        n += 1
    return n


def deep_rows(depth: np.ndarray, minor: int, deep_tail: float) -> np.ndarray:
    """bool[S]: samples whose EVERY site of the span (called depths [S, L]) is at or above deep_depth, so that N >= n >=
    deep holds everywhere and the read state is 'balanced' throughout: the exact condition for bypassing the read
    state (a low-depth island inside a high-mean sample keeps it)."""
    depth = np.asarray(depth)
    if depth.ndim != 2 or depth.shape[1] == 0:
        return np.zeros(depth.shape[0] if depth.ndim else 0, dtype=bool)
    return depth.min(axis=1) >= deep_depth(minor, deep_tail)


def table_depth(depth: np.ndarray, minor: int, parameters: ReadStateParameters) -> int:
    """Covering-depth range of the tables: the deepest called site + offsets + table_depth_margin (at least
    table_minimum_depth), capped at the deep bound + offsets + table_depth_margin (FIXED, so neither tables nor per-site
    work grow with sequencing depth). Every site below the deep bound is then represented at its physical covering
    depths n + o in laws, operators and allele likelihoods alike; a deep site (n >= deep_depth) is clamped only to
    covering depths where its class is 'balanced' to deep_tail, as its closed-form likelihood assumes."""
    extra = parameters.offsets + parameters.table_depth_margin
    cap = deep_depth(minor, parameters.deep_tail) + extra
    if depth.size == 0:
        return min(parameters.table_minimum_depth, cap)
    return int(min(max(parameters.table_minimum_depth, int(np.max(depth)) + extra), cap))


@dataclass(frozen=True)
class ReadTables:
    """Survival and depth bins of one sample batch with their depth and class tables, and the model's discretisation
    (offsets, the deep covering depth, the loss bound)."""

    survival_bin: np.ndarray        # int64[L] per gap (entry 0 unused)
    depth_bin: np.ndarray           # int64[S] per sample
    nmax: int
    minor: int
    depth: dict                     # survival bin -> float64[D, N, N'] (D depth bins)
    classes: dict                   # survival bin -> float64[D, N, N', C, C]
    offsets: int                    # loss offsets o in {0, ..., offsets - 1}
    deep: int                       # deep_depth(minor, deep_tail)
    maximum_loss: float
    # (survival bins int64[B] ascending, depth tables [D, B, N, N'], class tables [D, B, N, N', C, C]): the same tables
    # in one array each, depth value first (``depth`` and ``classes`` are views of them): a sample's tables are one
    # contiguous slice, read by the kernels without a copy (_stacked_tables)
    stacked: tuple | None = None


# Tables computed per item of build_tables' map (dimension-only: no effect on any value).
TABLE_CHUNK = 16


def _table_chunk(job):
    """[(depth_table, class_table)] of one chunk of (survival, arrivals, nmax, minor) keys sharing nmax and minor,
    with the survivor laws of their survivals (``rows``: survival -> survivor_laws). The convolutions and the class
    lumping run in nogil kernels (_arrival_kernel, _class_tables_kernel): the chunk's Python work is a fixed few
    calls, whatever its keys and depths. The tables are views of the chunk's arrays."""
    keys, rows = job
    nmax, minor = keys[0][2], keys[0][3]
    width, count = nmax + 1, len(keys)
    # One Poisson broadcast for each chunk: the one-homologue means of every key, then the two-homologue means.
    means = np.asarray([key[1] for key in keys])
    arrivals = poisson.pmf(np.arange(width), np.concatenate((means, 2.0 * means))[:, None])
    survivals = {f: i for i, f in enumerate(dict.fromkeys(key[0] for key in keys))}
    laws = np.stack([rows[f] for f in survivals])
    source = np.asarray([survivals[key[0]] for key in keys] * 2, dtype=np.int64)
    transitions = np.empty((2 * count, width, width))            # one homologue per key, then both homologues
    cls, n_classes, within, _ = _split_tables(nmax, minor)
    classes = np.empty((count, width, width, n_classes, n_classes))
    with bounded_team(2 * count * width):
        _arrival_kernel(laws, source, arrivals, transitions)
    with bounded_team(count * width):
        _class_tables_kernel(transitions[:count], cls, within, classes)
    return list(zip(transitions[count:], classes, strict=True))


def build_tables(gaps: np.ndarray, survival: ReadSurvival, covering: np.ndarray, nmax: int, minor: int,
                 parameters: ReadStateParameters, *, cache: dict | None = None, cache_bytes: int = 0,
                 map_work: Callable = map) -> ReadTables:
    """Tables for the batch: gaps (bp) between consecutive sites (entry 0 ignored), per-sample covering depth per
    homologue ``covering`` [S]. Each (survival bin, depth value) pair's tables are a function of (f, mu, nmax, minor)
    alone, and their survivor laws of (f, nmax): ``cache`` (a dict the caller owns, e.g. across a contig's sample
    batches) keeps both by those keys, and the missing ones are computed through ``map_work`` in chunks of
    TABLE_CHUNK keys (independent items; the same tables in any order or chunking). Each item writes its tables into
    the batch's stacked arrays at once (no second copy of the batch's tables); copies of the new tables then enter
    ``cache`` while its tables stay within ``cache_bytes``."""
    phi = survival.survival(gaps)
    log_turnover = np.log(np.clip(1.0 - phi, parameters.survival_floor, 1.0))
    survival_bin = np.rint(log_turnover / np.log(parameters.survival_step)).astype(np.int64)
    depth_bin = np.rint(covering / parameters.depth_step).astype(np.int64)
    depth_values = np.unique(depth_bin)
    depth_index = np.searchsorted(depth_values, depth_bin)
    keys = {}
    for b in np.unique(survival_bin[1:]):
        f = 1.0 - min(parameters.survival_step ** float(b), 1.0)
        arrivals = [max(j * parameters.depth_step, parameters.minimum_arrival_depth) * (1.0 - f) for j in depth_values]
        keys[b] = [(f, mu, int(nmax), int(minor)) for mu in arrivals]
    cache = {} if cache is None else cache
    n_classes = 2 * int(minor) + 3
    order = sorted(keys)
    big_depth = np.empty((len(depth_values), len(order), nmax + 1, nmax + 1))
    big_classes = np.empty((len(depth_values), len(order), nmax + 1, nmax + 1, n_classes, n_classes))
    depth_by_bin = {b: big_depth[:, i] for i, b in enumerate(order)}
    classes_by_bin = {b: big_classes[:, i] for i, b in enumerate(order)}
    slots = {}
    for b, row in keys.items():
        for j, key in enumerate(row):
            slots.setdefault(key, []).append((b, j))

    def place(key, depth, classes):
        for b, j in slots[key]:
            depth_by_bin[b][j] = depth
            classes_by_bin[b][j] = classes

    missing = [key for key in slots if key not in cache]
    cached = [key for key in slots if key in cache]
    survivals = list(dict.fromkeys(k[0] for k in missing if ("rows", k[0], int(nmax)) not in cache))
    for f, rows in zip(survivals, map_work(partial(survivor_laws, nmax=int(nmax)), survivals), strict=True):
        cache[("rows", f, int(nmax))] = rows

    # Cache admission follows key order and the byte bound. Workers copy only
    # the admitted entries, so the large array copies share the same team as
    # the warm slot copies and the cold table calculations.
    held = sum(sum(a.nbytes for a in pair) for key, pair in cache.items() if key[0] != "rows")
    table_bytes = (nmax + 1)**2 * (1 + n_classes**2) * big_depth.itemsize
    keep = set(missing[:max(0, (cache_bytes - held) // table_bytes)])

    def fill(job):
        cold, chunk = job
        if not cold:
            for key in chunk:
                place(key, *cache[key])
            return ()
        rows = {k[0]: cache[("rows", k[0], int(nmax))] for k in chunk}
        additions = []
        for key, (depth, classes) in zip(chunk, _table_chunk((chunk, rows)), strict=True):
            place(key, depth, classes)
            if key in keep:
                additions.append((key, (depth.copy(), classes.copy())))
        return additions

    threads = live_threads()
    work = []
    for cold, row in ((True, missing), (False, cached)):
        width = min(TABLE_CHUNK, max(1, len(row) // threads))
        work.extend((cold, row[i:i + width]) for i in range(0, len(row), width))
    # Some callers supply the lazy built-in map. Collect results before mutating
    # the shared cache so every map route has the same read-only worker inputs.
    for additions in list(map_work(fill, work)):
        cache.update(additions)
    return ReadTables(survival_bin, depth_index, nmax, minor, depth_by_bin, classes_by_bin, int(parameters.offsets),
                      deep_depth(minor, parameters.deep_tail), float(parameters.maximum_loss),
                      (np.array(order, dtype=np.int64), big_depth, big_classes))


def sample_tables(tables: ReadTables, sample: int) -> ReadTables:
    """The tables of one sample of the batch (its own depth value only, depth_bin 0): views, no copy."""
    d = int(tables.depth_bin[sample])
    stacked = None
    if tables.stacked is not None:
        keys, big_depth, big_classes = tables.stacked
        stacked = (keys, big_depth[d:d + 1], big_classes[d:d + 1])
    return replace(tables, depth_bin=np.zeros(1, dtype=np.int64),
                   depth={b: t[d:d + 1] for b, t in tables.depth.items()},
                   classes={b: t[d:d + 1] for b, t in tables.classes.items()}, stacked=stacked)


def _offset_inputs(depth: np.ndarray, covering: np.ndarray, loss: float, tables: ReadTables):
    """(covering depths int64[S, L, O] clamped to the tables, called-depth weights [S, L, O] rescaled per site, start
    law [S, O]) of the offset chain. Log space, rescaled per site by its largest offset weight: a called depth far
    above the tables (a spike of thousands of reads) would otherwise underflow every offset's PMF and zero the
    sample's whole chain; the per-site constant cancels in the normalised forward-backward. A site's weights are a
    function of its called depth alone: they are evaluated once per depth from 0 to the deepest site (the same
    elementwise SciPy and NumPy calls) and gathered per site in a Numba kernel (no per-site Python array work)."""
    loss = float(min(max(loss, 0.0), tables.maximum_loss))      # a probability (ReadSurvival.loss is already clipped)
    offsets = np.arange(tables.offsets)
    depth = np.ascontiguousarray(depth, dtype=np.int64)
    values = np.arange(int(depth.max()) + 1 if depth.size else 1)
    log_table = binom.logpmf(offsets[None, :], values[:, None] + offsets[None, :], loss)              # [D + 1, O]
    called_table = np.exp(log_table - log_table.max(axis=1, keepdims=True))
    covering_depth = np.empty(depth.shape + (tables.offsets,), dtype=np.int64)
    called = np.empty(depth.shape + (tables.offsets,))
    with bounded_team(depth.shape[0] * _site_blocks(depth.shape[1])):
        _offset_gather(depth, np.ascontiguousarray(called_table), int(tables.nmax - tables.offsets), covering_depth, called)
    log_start = poisson.logpmf(covering_depth[:, 0], 2.0 * covering[:, None]) + log_table[depth[:, 0]]
    start = np.exp(log_start - log_start.max(axis=1, keepdims=True))     # log space: no underflow at high depth
    return covering_depth, called, start / start.sum(axis=1, keepdims=True)


def offset_posteriors(depth: np.ndarray, covering: np.ndarray, loss: float, tables: ReadTables):
    """Depth-only forward-backward over the loss offsets of each sample (rows of ``depth`` [S, L], called depths;
    the reference form of offset_posteriors_fast). Returns (posterior [L, S, O], pairwise [L, S, O, O] of the gap
    l - 1 -> l (entry 0 unused), covering depths [S, L, O] clamped to the tables)."""
    n_samples, n_sites = depth.shape
    n_offsets = tables.offsets
    covering_depth, called, start = _offset_inputs(depth, covering, loss, tables)
    forward = np.empty((n_sites, n_samples, n_offsets))
    forward[0] = start

    def step(site):
        table = tables.depth[tables.survival_bin[site]]
        return table[tables.depth_bin[:, None, None], covering_depth[:, site - 1][:, :, None],
                     covering_depth[:, site][:, None, :]]

    for site in range(1, n_sites):
        value = np.einsum("so,sop->sp", forward[site - 1], step(site)) * called[:, site]
        total = value.sum(axis=1, keepdims=True)
        forward[site] = value / np.where(total > 0, total, 1.0)
    posterior = np.empty_like(forward)
    pairwise = np.zeros((n_sites, n_samples, n_offsets, n_offsets))
    backward = np.ones((n_samples, n_offsets))
    posterior[-1] = forward[-1]
    for site in range(n_sites - 1, 0, -1):
        transition = step(site)
        joint = forward[site - 1][:, :, None] * transition * (called[:, site] * backward)[:, None, :]
        total = joint.sum(axis=(1, 2), keepdims=True)
        pairwise[site] = joint / np.where(total > 0, total, 1.0)
        backward = np.einsum("sop,sp->so", transition, called[:, site] * backward)
        backward /= np.where(backward.sum(axis=1, keepdims=True) > 0, backward.sum(axis=1, keepdims=True), 1.0)
        both = forward[site - 1] * backward
        posterior[site - 1] = both / np.where(both.sum(axis=1, keepdims=True) > 0, both.sum(axis=1, keepdims=True), 1.0)
    return posterior, pairwise, covering_depth


def class_operators(pairwise: np.ndarray, covering_depth: np.ndarray, tables: ReadTables) -> np.ndarray:
    """Per-gap class operators [L, S, C, C] (entry 0: identity; the reference form of class_operators_fast):
    P(class' | class) over the gap, the class transitions at each pair of covering depths weighted by the joint law of
    the offsets AND the source class. The source class's law depends on its covering depth, so each offset pair
    (o, o') enters a row in proportion to pairwise(o, o') P(class | N = n + o) (the joint flow), and each row is
    divided by its source class's marginal; a source class with no mass takes the target law. The class law then
    travels exactly: site_law[l - 1] @ T_l = site_law[l]."""
    n_sites, n_samples = pairwise.shape[:2]
    cls, n_classes = split_class_map(tables.nmax, tables.minor)
    _, law = split_weights(tables.nmax, cls, n_classes)                  # P(class | N) [N, C]
    operators = np.empty((n_sites, n_samples, n_classes, n_classes))
    operators[0] = np.eye(n_classes)[None]
    for site in range(1, n_sites):
        table = tables.classes[tables.survival_bin[site]]
        block = table[tables.depth_bin[:, None, None], covering_depth[:, site - 1][:, :, None],
                      covering_depth[:, site][:, None, :]]
        source = law[covering_depth[:, site - 1]]                         # [S, O, C]
        flow = np.einsum("sab,sax,sabxy->sxy", pairwise[site], source, block)
        marginal = np.einsum("sab,sax->sx", pairwise[site], source)
        target = np.einsum("sab,sby->sy", pairwise[site], law[covering_depth[:, site]])
        reachable = marginal > 0
        operators[site] = np.where(reachable[:, :, None], flow / np.where(reachable, marginal, 1.0)[:, :, None],
                                   target[:, None, :])
    return operators


def site_laws(posterior: np.ndarray, covering_depth: np.ndarray, tables: ReadTables) -> np.ndarray:
    """[L, S, C] class law P(class | N) at every site averaged over its offset posterior (the law the GL-anchored read
    factor is normalised against)."""
    cls, n_classes = split_class_map(tables.nmax, tables.minor)
    _, law = split_weights(tables.nmax, cls, n_classes)
    return np.einsum("lso,slox->lsx", posterior, law[covering_depth])


def compose_operators(operators: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Operators between consecutive kept sites [L_kept, S, C, C]: the products of the per-gap operators over every
    skipped site (alleles at skipped sites unused; their depths informed the offsets). Entry 0: identity."""
    sites = np.flatnonzero(kept)
    n_samples, n_classes = operators.shape[1], operators.shape[2]
    out = np.empty((len(sites), n_samples, n_classes, n_classes))
    if len(sites) == 0:
        return out
    out[0] = np.eye(n_classes)[None]
    for index in range(1, len(sites)):
        product = operators[sites[index - 1] + 1]
        for site in range(sites[index - 1] + 2, sites[index] + 1):
            product = np.einsum("sxy,syz->sxz", product, operators[site])
        out[index] = product
    return out


def _conditional_offsets(called: np.ndarray, posterior: np.ndarray, tables: ReadTables):
    """(cls, within, conditional [S, L, O, C]): the split classes and within-class laws of the tables' depths, and
    the offsets weighted CONDITIONALLY on the class, P(o | class, depths) ~ posterior(o) P(class | N = n + o),
    normalised over o for each class (an offset at which the class is impossible gets no weight)."""
    cls, n_classes, within, class_law = _split_tables(tables.nmax, tables.minor)
    called = np.ascontiguousarray(called, dtype=np.int64)
    conditional = np.empty(called.shape + (posterior.shape[2], n_classes))                              # [S, L, O, C]
    with bounded_team(called.shape[0] * _site_blocks(called.shape[1])):
        _conditional_kernel(called, np.ascontiguousarray(posterior, dtype=np.float64), np.ascontiguousarray(class_law),
                            int(tables.nmax), conditional)
    return cls, within, conditional


def class_likelihoods(ref: np.ndarray, alt: np.ndarray, error, posterior: np.ndarray, tables: ReadTables) -> np.ndarray:
    """[S, L, 4, C] likelihood of each site's REF/ALT counts given the class and the ordered allele configuration
    (g1, g2) = 00, 01, 10, 11 of the two homologues, averaged over the class-conditional offset posterior: the class's
    splits (K1, N - K1) with their within-class law, the hypergeometric called split (c1 of the n called reads from
    homologue 1) and binomial reads with the per-read other-allele rate ``error`` (one rate, or one per sample). Not
    normalised (consumers take ratios per site). The reference form of class_likelihoods_fast."""
    called = (ref + alt).astype(np.int64)
    n_samples, n_sites = called.shape
    cls, within, conditional = _conditional_offsets(called, posterior, tables)
    n_classes = conditional.shape[3]
    error = np.broadcast_to(np.asarray(error, dtype=np.float64), (n_samples,))[:, None]     # per sample (or one rate)
    allele = (error, 1.0 - error)
    top = int(called.max()) if called.size else 0
    mixed = {}
    for g1 in (0, 1):
        for g2 in (0, 1):
            for c1 in range(top + 1):
                c2 = called - c1
                ok = c2 >= 0
                value = np.zeros((n_samples, n_sites))
                for j in range(c1 + 1):
                    rest = alt - j
                    good = ok & (rest >= 0) & (rest <= c2)
                    if not good.any():
                        continue
                    value += np.where(
                        good,
                        binom.pmf(j, c1, allele[g1])
                        * binom.pmf(np.where(good, rest, 0), np.where(ok, c2, 0), allele[g2]),
                        0.0,
                    )
                mixed[(g1, g2, c1)] = value
    out = np.zeros((n_samples, n_sites, 4, n_classes))
    rows = np.arange(n_samples)[:, None]
    columns = np.arange(n_sites)[None, :]
    for offset in range(tables.offsets):
        total = np.minimum(called + offset, tables.nmax)
        for first in range(int(total.max()) + 1 if total.size else 0):
            valid = first <= total
            if not valid.any():
                continue
            member = cls[total, np.minimum(first, total)]
            w = np.where(valid, within[total, np.minimum(first, total)], 0.0) * np.take_along_axis(
                conditional[:, :, offset, :], member[:, :, None], axis=2)[:, :, 0]
            for c1 in range(top + 1):
                with np.errstate(invalid="ignore", divide="ignore"):
                    draw = hypergeom.pmf(c1, np.maximum(total, 1), np.minimum(first, np.maximum(total, 1)),
                                         np.minimum(called, np.maximum(total, 1)))
                possible = valid & (c1 <= first) & (c1 <= called) & (called - c1 <= total - first)
                draw = np.where(np.isfinite(draw) & possible, draw, 0.0)
                draw = np.where(total == 0, (c1 == 0) * 1.0, draw)
                if not draw.any():
                    continue
                for config, (g1, g2) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                    np.add.at(out[:, :, config, :], (rows, columns, member), w * draw * mixed[(g1, g2, c1)])
    # deep sites (called depth >= the deep bound): balanced = the closed form Bin(a; n, (p_g1 + p_g2) / 2); the tail
    # class with the minor homologue holding exactly j reads = the split likelihood of (n - j, j) called reads, as the
    # kernels compute them
    deep = called >= tables.deep
    if deep.any():
        out[deep] = 0.0
        minor = tables.minor
        for config, (g1, g2) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            p1 = np.broadcast_to(allele[g1], called.shape)
            p2 = np.broadcast_to(allele[g2], called.shape)
            value = binom.pmf(alt, called, 0.5 * (p1 + p2))
            out[:, :, config, 0] = np.where(deep, value, out[:, :, config, 0])
            for j in range(minor + 1):
                # tail classes: homologue 2 (class 1 + j) or homologue 1 (class minor + 2 + j) holds exactly j reads
                for c, (big_p, small_p) in ((1 + j, (p1, p2)), (minor + 2 + j, (p2, p1))):
                    value = np.zeros(called.shape)
                    for r in range(j + 1):                   # r ALT among the j reads of the minor homologue
                        value += binom.pmf(r, j, small_p) * binom.pmf(alt - r, called - j, big_p)
                    out[:, :, config, c] = np.where(deep, value, out[:, :, config, c])
    return out


# Independent site intervals reuse one scratch space per work item. Grain affects scheduling only.
SITE_BLOCK = 128


def _site_blocks(n_sites):
    return (n_sites + SITE_BLOCK - 1) // SITE_BLOCK


# Exact argument types of the kernels' call sites (core.kernels audit).
_I64 = types.int64
_F64 = types.float64
_LOG_PAIR = types.UniTuple(types.float64, 2)
_I1 = types.Array(types.int64, 1, "C")
_I2 = types.Array(types.int64, 2, "C")
_I3 = types.Array(types.int64, 3, "C")
_F1 = types.Array(types.float64, 1, "C")
_F2 = types.Array(types.float64, 2, "C")
_F3 = types.Array(types.float64, 3, "C")
_F4 = types.Array(types.float64, 4, "C")
_F5 = types.Array(types.float64, 5, "C")
_F6 = types.Array(types.float64, 6, "C")
_OVERLOADS = {
    "_log_choose": ((_I64, _I64),),
    "_split_likelihood": ((_I64, _I64, _I64, _F64, _F64, _F64, _F64),),
    # the sample index is a prange index: int64 in one calling kernel, uint64 in the other
    "_deep_site": tuple((_F4, index, _I64, _I64, _I64, _LOG_PAIR, _LOG_PAIR, _I64) for index in (_I64, types.uint64)),
    "_class_likelihoods_kernel": ((_I2, _I2, _F1, _F4, _I2, _F2, _I64, _I64, _F4),),
    "_class_tables_kernel": ((_F3, _I2, _F2, _F5),),
    "_arrival_kernel": ((_F3, _I1, _F2, _F3),),
    "_offset_messages": ((_F2, _F3, _I3, _I1, _I1, _F4, _I2, _F3, _F3),),
    "_offset_edges": ((_F3, _I3, _I1, _I1, _F4, _I2, _F3, _F3, _F3, _F4, _F6, _F2, _F4),),
    "_site_laws_kernel": ((_F3, _I3, _I1, _F2, _F3),),
    "_offset_support": ((_F3, _I64, _F64, _I2),),
    "_operator_kernel": ((_F4, _I3, _I1, _I1, _F6, _F2, _F4),),
    "_class_likelihood_slots": ((_I2, _I2, _I2, _I64),),
    "_class_likelihood_bases": ((_I1, _I1, _F1, _I2, _F2, _I64, _F4),),
    "_class_likelihoods_blocks": ((_I2, _I2, _F1, _F3, _F2, _I2, _F4, _I64, _F4),),
    "_compose_kernel": ((_F4, _I1, _F4),),
    "_offset_gather": ((_I2, _F2, _I64, _I3, _F3),),
    "_class_flow": ((_F2, _I1, _I1, _I64, _I64, _F4, _F2, _F2, _F1, _F1, _F2),),
    "_conditional_kernel": ((_I2, _F3, _F2, _I64, _F4),),
}


@kernel(overloads=_OVERLOADS["_class_tables_kernel"], cache=True, parallel=True, nogil=True)
def _class_tables_kernel(ones, cls, within, out):
    """Lump each homologue transition into classes, keeping the split sums local to a table row."""
    width, n_classes = ones.shape[1], out.shape[4]
    for task in prange(len(ones) * width):
        table_index, total = task // width, task % width
        table = ones[table_index]
        flow = np.empty(n_classes)
        for next_total in range(width):
            result = out[table_index, total, next_total]
            for x in range(n_classes):
                for y in range(n_classes):
                    result[x, y] = 0.0
            for next_first in range(next_total + 1):
                next_second = next_total - next_first
                for x in range(n_classes):
                    flow[x] = 0.0
                for first in range(total + 1):
                    joint = table[first, next_first] * table[total - first, next_second]
                    flow[cls[total, first]] += within[total, first] * joint
                target = cls[next_total, next_first]
                for x in range(n_classes):
                    result[x, target] += flow[x]
            for x in range(n_classes):
                mass = 0.0
                for y in range(n_classes):
                    mass += result[x, y]
                for y in range(n_classes):
                    result[x, y] = result[x, y] / mass if mass > 0.0 else 1.0 / n_classes


@kernel(overloads=_OVERLOADS["_arrival_kernel"], cache=True, parallel=True, nogil=True)
def _arrival_kernel(laws, source, arrivals, out):
    """_arrival_table for every row of ``arrivals`` [A, N] at once: out[a, k, m] = sum over j <= min(k, m) of
    laws[source[a], k, j] arrivals[a, m - j], the first N terms of the convolution of covering depth k's survivor law
    (survivor_laws) with the arrival law, summed over the survivors in ascending order."""
    width = out.shape[1]
    for task in prange(out.shape[0] * width):
        a, k = task // width, task % width
        law = laws[source[a], k]
        new = arrivals[a]
        row = out[a, k]
        for m in range(width):
            total = 0.0
            for j in range(min(k, m) + 1):
                total += law[j] * new[m - j]
            row[m] = total


@kernel(overloads=_OVERLOADS["_site_laws_kernel"], cache=True, parallel=True, nogil=True)
def _site_laws_kernel(posterior, covering_depth, sites, law, out):
    n_samples = covering_depth.shape[0]
    blocks = (len(sites) + SITE_BLOCK - 1) // SITE_BLOCK
    for task in prange(n_samples * blocks):
        s, block = task // blocks, task % blocks
        for i in range(block * SITE_BLOCK, min(len(sites), (block + 1) * SITE_BLOCK)):
            l = sites[i]
            for c in range(law.shape[1]):
                total = 0.0
                for o in range(posterior.shape[2]):
                    total += posterior[l, s, o] * law[covering_depth[s, l, o], c]
                out[i, s, c] = total


def site_laws_fast(posterior, covering_depth, sites, tables):
    """Class laws at selected sites, contracting the offset posterior in offset order."""
    _, n_classes, _, law = _split_tables(tables.nmax, tables.minor)
    n_samples = covering_depth.shape[0]
    out = np.empty((len(sites), n_samples, n_classes))
    with bounded_team(n_samples * _site_blocks(len(sites))):
        _site_laws_kernel(posterior, np.ascontiguousarray(covering_depth), sites, law, out)
    return out


@kernel(overloads=_OVERLOADS["_offset_gather"], cache=True, parallel=True, nogil=True)
def _offset_gather(depth, called_table, cap, covering_depth, called):
    """_offset_inputs' per-site arrays: covering_depth[s, l, o] = min(depth, cap) + o, called[s, l, o] =
    called_table[depth[s, l], o] (the row of the site's called depth; copies of the table's values)."""
    n_samples, n_sites = depth.shape
    n_offsets = called_table.shape[1]
    blocks = (n_sites + SITE_BLOCK - 1) // SITE_BLOCK
    for task in prange(n_samples * blocks):
        s, block = task // blocks, task % blocks
        for l in range(block * SITE_BLOCK, min(n_sites, (block + 1) * SITE_BLOCK)):
            n = depth[s, l]
            clamped = n if n < cap else cap
            for o in range(n_offsets):
                covering_depth[s, l, o] = clamped + o
                called[s, l, o] = called_table[n, o]


@kernel(overloads=_OVERLOADS["_conditional_kernel"], cache=True, parallel=True, nogil=True)
def _conditional_kernel(called, posterior, class_law, nmax, out):
    """_conditional_offsets' weights: out[s, l, o, c] = posterior[l, s, o] class_law[min(called + o, nmax), c], each
    class's weights divided by their sum over o where it is positive (the NumPy expression's products, its ordered
    sum over the offsets and its division)."""
    n_samples, n_sites = called.shape
    n_offsets = posterior.shape[2]
    n_classes = class_law.shape[1]
    blocks = (n_sites + SITE_BLOCK - 1) // SITE_BLOCK
    for task in prange(n_samples * blocks):
        s, block = task // blocks, task % blocks
        for l in range(block * SITE_BLOCK, min(n_sites, (block + 1) * SITE_BLOCK)):
            for o in range(n_offsets):
                n = called[s, l] + o
                if n > nmax:
                    n = nmax
                p = posterior[l, s, o]
                for c in range(n_classes):
                    out[s, l, o, c] = p * class_law[n, c]
            for c in range(n_classes):
                total = out[s, l, 0, c]
                for o in range(1, n_offsets):
                    total += out[s, l, o, c]
                if total > 0.0:
                    for o in range(n_offsets):
                        out[s, l, o, c] /= total


@kernel(overloads=_OVERLOADS["_log_choose"], cache=True, nogil=True)
def _log_choose(n, k):
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


@kernel(overloads=_OVERLOADS["_split_likelihood"], cache=True, nogil=True)
def _split_likelihood(n1, n2, a, lp1, lq1, lp2, lq2):
    """P(a ALT among n1 reads of homologue 1 and n2 of homologue 2) = sum_k Bin(k; n1, p1) Bin(a - k; n2, p2): at a deep
    site one of n1, n2 is a tail count j <= minor, so the sum has at most j + 1 terms (constant work at any depth)."""
    acc = 0.0
    for k in range(max(0, a - n2), min(n1, a) + 1):
        r = a - k
        acc += math.exp(_log_choose(n1, k) + k * lp1 + (n1 - k) * lq1 + _log_choose(n2, r) + r * lp2 + (n2 - r) * lq2)
    return acc


@kernel(overloads=_OVERLOADS["_deep_site"], cache=True, nogil=True)
def _deep_site(out, s, l, n, a, log_p, log_q, n_classes):
    """Class likelihoods at a deep site (n >= the deep bound): balanced = the closed form Bin(a; n, (p1 + p2) / 2); the
    tail class with homologue 2 (resp. 1) holding exactly j <= minor reads = the split likelihood of (n - j, j) (resp.
    (j, n - j)) called reads, offsets not modelled at deep sites. A homozygous configuration gets the same value in
    every class (neutral, whatever class the read state carries in)."""
    minor = (n_classes - 3) // 2
    for g1 in range(2):
        for g2 in range(2):
            config = g1 * 2 + g2
            pa = 0.5 * (math.exp(log_p[g1]) + math.exp(log_p[g2]))
            out[s, l, config, 0] += math.exp(_log_choose(n, a) + a * math.log(pa) + (n - a) * math.log(1.0 - pa))
            for j in range(minor + 1):
                out[s, l, config, 1 + j] += _split_likelihood(n - j, j, a, log_p[g1], log_q[g1], log_p[g2], log_q[g2])
                out[s, l, config, minor + 2 + j] += _split_likelihood(j, n - j, a, log_p[g1], log_q[g1], log_p[g2],
                                                                      log_q[g2])


@kernel(overloads=_OVERLOADS["_class_likelihoods_kernel"], cache=True, parallel=True, nogil=True)
def _class_likelihoods_kernel(ref, alt, error, conditional, cls, within, nmax, deep, out):
    """out[S, L, 4, C]: class_likelihoods' sum, per sample (prange) and site: for each offset o, covering depth
    N = min(n + o, nmax), each split K1 of N with its class and within-class law, the hypergeometric called split c1
    and the configuration's allele likelihood of the counts (c1 reads of homologue 1's allele, n - c1 of homologue
    2's), weighted by the class-conditional offset weight."""
    n_samples, n_sites = ref.shape
    n_offsets = conditional.shape[2]
    for s in prange(n_samples):
        e = error[s]
        log_p = (math.log(e), math.log(1.0 - e))          # P(ALT read | allele 0), P(ALT read | allele 1)
        log_q = (math.log(1.0 - e), math.log(e))          # P(REF read | allele 0), P(REF read | allele 1)
        for l in range(n_sites):
            a = alt[s, l]
            n = ref[s, l] + a
            if n >= deep:
                _deep_site(out, s, l, n, a, log_p, log_q, out.shape[3])
                continue
            # mixed[g1, g2, c1] = sum_j Bin(j; c1, p_g1) Bin(a - j; n - c1, p_g2)
            mixed = np.zeros((2, 2, n + 1))
            for c1 in range(n + 1):
                c2 = n - c1
                for g1 in range(2):
                    for g2 in range(2):
                        acc = 0.0
                        for j in range(max(0, a - c2), min(c1, a) + 1):
                            r = a - j
                            acc += math.exp(_log_choose(c1, j) + j * log_p[g1] + (c1 - j) * log_q[g1]
                                            + _log_choose(c2, r) + r * log_p[g2] + (c2 - r) * log_q[g2])
                        mixed[g1, g2, c1] = acc
            for o in range(n_offsets):
                total = min(n + o, nmax)
                if n > total:                 # called depth beyond the tables' range: no split (as the reference)
                    continue
                if total == 0:
                    member = cls[0, 0]
                    w = conditional[s, l, o, member]
                    for config in range(4):
                        out[s, l, config, member] += w * mixed[config // 2, config % 2, 0]
                    continue
                for first in range(total + 1):
                    member = cls[total, first]
                    w = within[total, first] * conditional[s, l, o, member]
                    if w == 0.0:
                        continue
                    second = total - first
                    log_all = _log_choose(total, n)
                    for c1 in range(max(0, n - second), min(first, n) + 1):
                        draw = math.exp(_log_choose(first, c1) + _log_choose(second, n - c1) - log_all)
                        for config in range(4):
                            out[s, l, config, member] += w * draw * mixed[config // 2, config % 2, c1]


def _stacked_tables(tables: ReadTables, depth: bool = True, classes: bool = True):
    """(table index per site int64[L] (entry 0 unused), depth tables [D, B, N, N'], class tables [D, B, N, N', C, C]):
    the per-survival-bin tables as arrays for the kernels; only the requested ones are stacked (None otherwise: the
    class tables of a batch with many survival bins and depth values are gigabytes)."""
    keys = tables.stacked[0] if tables.stacked is not None else np.array(sorted(tables.depth), dtype=np.int64)
    index = np.clip(np.searchsorted(keys, tables.survival_bin), 0, max(len(keys) - 1, 0)).astype(np.int64)
    index[0] = 0
    if tables.stacked is not None:
        _, big_depth, big_classes = tables.stacked
        return index, big_depth if depth else None, big_classes if classes else None
    return (index, np.ascontiguousarray(np.stack([tables.depth[k] for k in keys], axis=1)) if depth else None,
            np.ascontiguousarray(np.stack([tables.classes[k] for k in keys], axis=1)) if classes else None)


@kernel(overloads=_OVERLOADS["_class_flow"], cache=True, nogil=True)
def _class_flow(joint, left, right, previous, current, classes, law, flow, marg, targ, out):
    """Mix the class transition over one edge's joint depth law, in offset order."""
    n_classes = law.shape[1]
    flow[:, :] = 0.0
    marg[:] = 0.0
    targ[:] = 0.0
    for o in range(previous):
        n0 = left[o]
        for q in range(current):
            p = joint[o, q]
            if p == 0.0:
                continue
            n1 = right[q]
            for y in range(n_classes):
                targ[y] += p * law[n1, y]
            for x in range(n_classes):
                wx = p * law[n0, x]
                if wx == 0.0:
                    continue
                marg[x] += wx
                for y in range(n_classes):
                    flow[x, y] += wx * classes[n0, n1, x, y]
    for x in range(n_classes):
        for y in range(n_classes):
            out[x, y] = flow[x, y] / marg[x] if marg[x] > 0.0 else targ[y]


@kernel(overloads=_OVERLOADS["_offset_messages"], cache=True, parallel=True, nogil=True)
def _offset_messages(start, called, cdepth, sindex, dbin, dtab, support, fwd, bwd):
    """Two independent directions per sample. Each direction retains its ordered depth recurrence."""
    n_samples, n_sites, n_off = fwd.shape
    full = support.shape[0] == 0
    for task in prange(2 * n_samples):
        s = task // 2
        d = dbin[s]
        if task % 2 == 0:
            previous = n_off if full else support[s, 0]
            for o in range(previous):
                fwd[s, 0, o] = start[s, o]
            if previous < start.shape[1]:
                total = fwd[s, 0, :previous].sum()
                if total > 0.0:
                    fwd[s, 0, :previous] /= total
            for l in range(1, n_sites):
                trans = dtab[d, sindex[l]]
                current = n_off if full else support[s, l]
                total = 0.0
                for q in range(current):
                    acc = 0.0
                    for o in range(previous):
                        acc += fwd[s, l - 1, o] * trans[cdepth[s, l - 1, o], cdepth[s, l, q]]
                    value = acc * called[s, l, q]
                    fwd[s, l, q] = value
                    total += value
                if total > 0.0:
                    for q in range(current):
                        fwd[s, l, q] /= total
                previous = current
        else:
            current = n_off if full else support[s, n_sites - 1]
            bwd[s, n_sites - 1, :current] = 1.0
            for l in range(n_sites - 1, 0, -1):
                trans = dtab[d, sindex[l]]
                previous = n_off if full else support[s, l - 1]
                total = 0.0
                for o in range(previous):
                    acc = 0.0
                    for q in range(current):
                        acc += trans[cdepth[s, l - 1, o], cdepth[s, l, q]] * called[s, l, q] * bwd[s, l, q]
                    bwd[s, l - 1, o] = acc
                    total += acc
                if total > 0.0:
                    for o in range(previous):
                        bwd[s, l - 1, o] /= total
                current = previous


@kernel(overloads=_OVERLOADS["_offset_edges"], cache=True, parallel=True, nogil=True)
def _offset_edges(called, cdepth, sindex, dbin, dtab, support, fwd, bwd, posterior, pairwise, ctab, law, operators):
    """Site marginals and edge laws from the saved directional messages."""
    n_samples, n_sites, n_off = fwd.shape
    full = support.shape[0] == 0
    store_pairs = pairwise.shape[0] != 0
    build_operators = operators.shape[0] != 0
    n_classes = law.shape[1]
    blocks = (n_sites + SITE_BLOCK - 1) // SITE_BLOCK
    for task in prange(n_samples * blocks):
        s, block = task // blocks, task % blocks
        d = dbin[s]
        edge_width = n_off if build_operators else 0
        edge = np.empty((edge_width, edge_width))
        flow = np.empty((n_classes, n_classes))
        marg = np.empty(n_classes)
        targ = np.empty(n_classes)
        for l in range(block * SITE_BLOCK, min(n_sites, (block + 1) * SITE_BLOCK)):
            current = n_off if full else support[s, l]
            if l == n_sites - 1:
                posterior[l, s, :current] = fwd[s, l, :current]
            else:
                total = 0.0
                for o in range(current):
                    total += fwd[s, l, o] * bwd[s, l, o]
                for o in range(current):
                    product = fwd[s, l, o] * bwd[s, l, o]
                    posterior[l, s, o] = product / total if total > 0.0 else product
            posterior[l, s, current:] = 0.0
            if l == 0:
                if build_operators:
                    for x in range(n_classes):
                        for y in range(n_classes):
                            operators[0, s, x, y] = 1.0 if x == y else 0.0
                continue
            if store_pairs or build_operators:
                b = sindex[l]
                trans = dtab[d, b]
                previous = n_off if full else support[s, l - 1]
                joint = edge if build_operators else pairwise[l, s]
                total = 0.0
                for o in range(previous):
                    for q in range(current):
                        value = (fwd[s, l - 1, o] * trans[cdepth[s, l - 1, o], cdepth[s, l, q]]
                                 * called[s, l, q] * bwd[s, l, q])
                        joint[o, q] = value
                        total += value
                if total > 0.0:
                    for o in range(previous):
                        for q in range(current):
                            joint[o, q] /= total
                if build_operators:
                    _class_flow(joint, cdepth[s, l - 1], cdepth[s, l], previous, current, ctab[d, b],
                                law, flow, marg, targ, operators[l, s])


@kernel(overloads=_OVERLOADS["_offset_support"], cache=True, parallel=True, nogil=True)
def _offset_support(posterior, minimum, span_tolerance, support):
    """Exclude a tail of at most span_tolerance / n_sites at each site of the pilot law."""
    n_sites, n_samples, pilot = posterior.shape
    local_tolerance = span_tolerance / n_sites
    blocks = (n_sites + SITE_BLOCK - 1) // SITE_BLOCK
    for task in prange(n_samples * blocks):
        s, block = task // blocks, task % blocks
        for site in range(block * SITE_BLOCK, min(n_sites, (block + 1) * SITE_BLOCK)):
            width = pilot
            tail = 0.0
            while width > minimum:
                tail += posterior[site, s, width - 1]
                if tail > local_tolerance:
                    break
                width -= 1
            support[s, site] = width


def offset_pilot_width(covering: np.ndarray, loss: float, parameters: ReadStateParameters) -> int:
    """Finite depth pilot, fixed by the library rather than by execution batches."""
    if loss == 0.0 or not covering.size:
        return parameters.offsets
    lost_mean = 2.0 * float(covering.max()) * loss
    width = int(poisson.isf(parameters.loss_pilot_tail, lost_mean)) + parameters.loss_pilot_extra_offsets
    return max(parameters.offsets,
               min(parameters.loss_pilot_maximum_offsets, max(parameters.loss_pilot_minimum_offsets, width)))


def _fit_offsets(depth, covering, loss, tables, minimum_offsets, span_tail, *, build_operators):
    n_samples, n_sites = depth.shape
    adaptive = minimum_offsets < tables.offsets
    covering_depth, called, start = _offset_inputs(depth, covering, loss, tables)
    sindex, dtab, ctab = _stacked_tables(tables, classes=build_operators)
    inputs = (np.ascontiguousarray(start), np.ascontiguousarray(called), np.ascontiguousarray(covering_depth),
              sindex, np.ascontiguousarray(tables.depth_bin, dtype=np.int64), dtab)
    support = np.empty((0, 0), dtype=np.int64)
    no_edges = np.empty((0, 0, 0, 0))
    no_classes = np.empty((0, 0, 0, 0, 0, 0))
    no_law = np.empty((0, 0))
    def fit(support, width, pairwise, class_tables, law, operators):
        fwd = np.empty((n_samples, n_sites, width))
        bwd = np.empty_like(fwd)
        posterior = np.empty((n_sites, n_samples, width))
        with bounded_team(2 * n_samples):
            _offset_messages(*inputs, support, fwd, bwd)
        with bounded_team(n_samples * _site_blocks(n_sites)):
            _offset_edges(*inputs[1:], support, fwd, bwd, posterior, pairwise, class_tables, law, operators)
        return posterior

    if adaptive:
        pilot = fit(support, tables.offsets, no_edges, no_classes, no_law, no_edges)
        support = np.empty((n_samples, n_sites), dtype=np.int64)
        with bounded_team(n_samples * _site_blocks(n_sites)):
            _offset_support(pilot, minimum_offsets, span_tail, support)
        del pilot
    if build_operators:
        _, n_classes, _, law = _split_tables(tables.nmax, tables.minor)
        operators = np.empty((n_sites, n_samples, n_classes, n_classes))
        # Depths still use the pilot's table clip. The subsequent kernels need only the retained offsets.
        width = int(support.max()) if adaptive else tables.offsets
        posterior = fit(support, width, no_edges, ctab, law, operators)
        return posterior, operators, covering_depth[:, :, :width]
    pairwise = np.zeros((n_sites, n_samples, tables.offsets, tables.offsets))
    posterior = fit(support, tables.offsets, pairwise, no_classes, no_law, no_edges)
    return posterior, pairwise, covering_depth


def offset_posteriors_fast(depth: np.ndarray, covering: np.ndarray, loss: float, tables: ReadTables, *,
                           adaptive: ReadStateParameters | None = None):
    """Depth-only forward-backward, with optional pilot conditioning; materialize the pair laws for comparison."""
    if depth.shape[1] == 0 or (depth.shape[1] == 1 and (adaptive is None or adaptive.offsets >= tables.offsets)):
        return offset_posteriors(depth, covering, loss, tables)
    minimum = tables.offsets if adaptive is None else adaptive.offsets
    return _fit_offsets(depth, covering, loss, tables, minimum,
                        0.0 if adaptive is None else adaptive.loss_span_tail, build_operators=False)


def offset_operators_fast(depth: np.ndarray, covering: np.ndarray, loss: float, tables: ReadTables):
    """Offset marginals and class operators using the fixed offset support."""
    return _fit_offsets(depth, covering, loss, tables, tables.offsets, 0.0, build_operators=True)


def adaptive_offset_operators(depth: np.ndarray, covering: np.ndarray, loss: float, tables: ReadTables,
                              parameters: ReadStateParameters):
    """Condition the pilot depth path on site supports and consume one joint edge law at a time.

    The pilot only needs site marginals to select the supports. Recompute the restricted joint path law before
    mixing its class operators, then return those operators, the retained marginals and their covering depths.
    """
    return _fit_offsets(depth, covering, loss, tables, parameters.offsets, parameters.loss_span_tail,
                        build_operators=True)


@kernel(overloads=_OVERLOADS["_operator_kernel"], cache=True, parallel=True, nogil=True)
def _operator_kernel(pairwise, cdepth, sindex, dbin, ctab, law, out):
    """class_operators' joint flow over independent sample and site blocks."""
    n_sites, n_samples, n_off, _ = pairwise.shape
    n_classes = law.shape[1]
    blocks = (n_sites + SITE_BLOCK - 1) // SITE_BLOCK
    for task in prange(n_samples * blocks):
        s, block = task // blocks, task % blocks
        d = dbin[s]
        flow = np.empty((n_classes, n_classes))
        marg = np.empty(n_classes)
        targ = np.empty(n_classes)
        if block == 0:
            for x in range(n_classes):
                for y in range(n_classes):
                    out[0, s, x, y] = 1.0 if x == y else 0.0
        for l in range(max(1, block * SITE_BLOCK), min(n_sites, (block + 1) * SITE_BLOCK)):
            b = sindex[l]
            _class_flow(pairwise[l, s], cdepth[s, l - 1], cdepth[s, l], n_off, n_off, ctab[d, b],
                        law, flow, marg, targ, out[l, s])


def class_operators_fast(pairwise: np.ndarray, covering_depth: np.ndarray, tables: ReadTables) -> np.ndarray:
    """class_operators (same values) parallel over sample and site blocks."""
    n_sites, n_samples = pairwise.shape[:2]
    if n_sites < 2:
        return class_operators(pairwise, covering_depth, tables)
    _, n_classes, _, law = _split_tables(tables.nmax, tables.minor)
    sindex, _, ctab = _stacked_tables(tables, depth=False)
    out = np.empty((n_sites, n_samples, n_classes, n_classes))
    with bounded_team(n_samples * _site_blocks(n_sites)):
        _operator_kernel(np.ascontiguousarray(pairwise), np.ascontiguousarray(covering_depth, dtype=np.int64), sindex,
                         np.ascontiguousarray(tables.depth_bin, dtype=np.int64), ctab, np.ascontiguousarray(law), out)
    return out


@kernel(overloads=_OVERLOADS["_class_likelihood_slots"], cache=True, parallel=True, nogil=True)
def _class_likelihood_slots(called, alt, slots, deep):
    """Mark the distinct shallow count pairs in each sample."""
    n_samples, n_sites = called.shape
    for s in prange(n_samples):
        for l in range(n_sites):
            n = called[s, l]
            if n < deep:
                slots[s, n * deep + alt[s, l]] = 1


@kernel(overloads=_OVERLOADS["_class_likelihood_bases"], cache=True, parallel=True, nogil=True)
def _class_likelihood_bases(rows, pairs, error, cls, within, deep, base):
    """Tabulate split sums once per distinct (sample, called depth, ALT count), in the usual split order."""
    n_offsets = base.shape[1]
    nmax = cls.shape[0] - 1
    for k in prange(len(rows)):
        s = rows[k]
        n, a = pairs[k] // deep, pairs[k] % deep
        e = error[s]
        log_p = (math.log(e), math.log(1.0 - e))
        log_q = (math.log(1.0 - e), math.log(e))
        mixed = np.zeros((2, 2, n + 1))
        for c1 in range(n + 1):
            c2 = n - c1
            for g1 in range(2):
                for g2 in range(2):
                    acc = 0.0
                    for j in range(max(0, a - c2), min(c1, a) + 1):
                        r = a - j
                        acc += math.exp(_log_choose(c1, j) + j * log_p[g1] + (c1 - j) * log_q[g1]
                                        + _log_choose(c2, r) + r * log_p[g2] + (c2 - r) * log_q[g2])
                    mixed[g1, g2, c1] = acc
        for o in range(n_offsets):
            total = min(n + o, nmax)
            if n > total:
                continue
            if total == 0:
                member = cls[0, 0]
                for config in range(4):
                    base[k, o, config, member] += mixed[config // 2, config % 2, 0]
                continue
            log_all = _log_choose(total, n)
            for first in range(total + 1):
                member = cls[total, first]
                w = within[total, first]
                if w == 0.0:
                    continue
                second = total - first
                for c1 in range(max(0, n - second), min(first, n) + 1):
                    draw = math.exp(_log_choose(first, c1) + _log_choose(second, n - c1) - log_all)
                    for config in range(4):
                        base[k, o, config, member] += w * draw * mixed[config // 2, config % 2, c1]


@kernel(overloads=_OVERLOADS["_class_likelihoods_blocks"], cache=True, parallel=True, nogil=True)
def _class_likelihoods_blocks(called, alt, error, posterior, law, slots, base, deep, out):
    """Normalise a site's class-conditional offsets and contract with its tabulated split likelihoods."""
    n_samples, n_sites = called.shape
    n_offsets, n_classes = posterior.shape[2], law.shape[1]
    nmax = law.shape[0] - 1
    blocks = (n_sites + SITE_BLOCK - 1) // SITE_BLOCK
    for task in prange(n_samples * blocks):
        s, block = task // blocks, task % blocks
        e = error[s]
        log_p = (math.log(e), math.log(1.0 - e))
        log_q = (math.log(1.0 - e), math.log(e))
        weights = np.empty((n_offsets, n_classes))
        for l in range(block * SITE_BLOCK, min(n_sites, (block + 1) * SITE_BLOCK)):
            a, n = alt[s, l], called[s, l]
            for g in range(4):
                for c in range(n_classes):
                    out[s, l, g, c] = 0.0
            if n >= deep:
                _deep_site(out, s, l, n, a, log_p, log_q, n_classes)
                continue
            k = slots[s, n * deep + a]
            for o in range(n_offsets):
                for c in range(n_classes):
                    weights[o, c] = posterior[l, s, o] * law[min(n + o, nmax), c]
            for c in range(n_classes):
                total = weights[0, c]
                for o in range(1, n_offsets):
                    total += weights[o, c]
                if total > 0.0:
                    for o in range(n_offsets):
                        weights[o, c] /= total
            for o in range(n_offsets):
                for g in range(4):
                    for c in range(n_classes):
                        out[s, l, g, c] += weights[o, c] * base[k, o, g, c]


@kernel(overloads=_OVERLOADS["_compose_kernel"], cache=True, parallel=True, nogil=True)
def _compose_kernel(operators, sites, out):
    """compose_operators' products over independent sample and kept-site blocks: out[index] = operators[sites[index - 1] + 1] @ ... @
    operators[sites[index]], each matrix product summed over the inner index in order (as the reference's einsum)."""
    n_samples, n_classes = operators.shape[1], operators.shape[2]
    blocks = (len(sites) + SITE_BLOCK - 1) // SITE_BLOCK
    for task in prange(n_samples * blocks):
        s, block = task // blocks, task % blocks
        product = np.empty((n_classes, n_classes))
        work = np.empty((n_classes, n_classes))
        if block == 0:
            for i in range(n_classes):
                for j in range(n_classes):
                    out[0, s, i, j] = 1.0 if i == j else 0.0
        for index in range(max(1, block * SITE_BLOCK), min(len(sites), (block + 1) * SITE_BLOCK)):
            for i in range(n_classes):
                for j in range(n_classes):
                    product[i, j] = operators[sites[index - 1] + 1, s, i, j]
            for site in range(sites[index - 1] + 2, sites[index] + 1):
                for i in range(n_classes):
                    for k in range(n_classes):
                        acc = 0.0
                        for y in range(n_classes):
                            acc += product[i, y] * operators[site, s, y, k]
                        work[i, k] = acc
                for i in range(n_classes):
                    for k in range(n_classes):
                        product[i, k] = work[i, k]
            for i in range(n_classes):
                for j in range(n_classes):
                    out[index, s, i, j] = product[i, j]


def compose_operators_fast(operators: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Compose kept-site intervals in parallel, preserving the products and their summation order."""
    sites = np.flatnonzero(kept).astype(np.int64)
    n_samples, n_classes = operators.shape[1], operators.shape[2]
    out = np.empty((len(sites), n_samples, n_classes, n_classes))
    if len(sites) == 0:
        return out
    with bounded_team(n_samples * _site_blocks(len(sites))):
        _compose_kernel(np.ascontiguousarray(operators, dtype=np.float64), sites, out)
    return out


def _likelihood_inputs(ref, alt, error, posterior, tables):
    called = (ref + alt).astype(np.int64)
    cls, within, conditional = _conditional_offsets(called, posterior, tables)
    error = np.array(np.broadcast_to(np.asarray(error, dtype=np.float64), (called.shape[0],)))     # a writable copy
    out = np.zeros((called.shape[0], called.shape[1], 4, conditional.shape[3]))
    return (np.ascontiguousarray(ref, dtype=np.int64), np.ascontiguousarray(alt, dtype=np.int64), error,
            np.ascontiguousarray(conditional), cls, within, out)


def class_likelihoods_fast(ref, alt, error, posterior, tables):
    """Tabulate the split sums over observed count pairs, then score independent sample and site blocks."""
    called = np.ascontiguousarray(ref + alt, dtype=np.int64)
    alt = np.ascontiguousarray(alt, dtype=np.int64)
    n_samples, n_sites = called.shape
    error = np.array(np.broadcast_to(np.asarray(error, dtype=np.float64), (n_samples,)))
    cls, n_classes, within, law = _split_tables(tables.nmax, tables.minor)
    deep = int(tables.deep)
    slots = np.zeros((n_samples, deep * deep), dtype=np.int64)
    with bounded_team(n_samples):
        _class_likelihood_slots(called, alt, slots, deep)
    rows, pairs = np.nonzero(slots)
    slots[rows, pairs] = np.arange(len(rows))
    bases = np.zeros((len(rows), posterior.shape[2], 4, n_classes))
    if len(rows):
        with bounded_team(len(rows)):
            _class_likelihood_bases(np.ascontiguousarray(rows), np.ascontiguousarray(pairs), error, cls, within,
                                    deep, bases)
    out = np.empty((n_samples, n_sites, 4, n_classes))
    with bounded_team(n_samples * _site_blocks(n_sites)):
        _class_likelihoods_blocks(called, alt, error, np.ascontiguousarray(posterior), law, slots, bases, deep, out)
    return out


def class_likelihoods_sitewise(ref, alt, error, posterior, tables):
    """class_likelihoods_fast's per-site kernel (_class_likelihoods_kernel), kept as the check of the factored one."""
    ref, alt, error, conditional, cls, within, out = _likelihood_inputs(ref, alt, error, posterior, tables)
    _class_likelihoods_kernel(ref, alt, error, conditional, cls, within, int(tables.nmax), int(tables.deep), out)
    return out


# ------------------------------------------------------------------------------------------------ libraries


def _correlation_curves(depth, positions, lags, *, map_work=map):
    """[S, lags] each sample's correlation of called depths (NaN if a lag has no pairs).

    Pair indices are shared across the sample blocks. A block keeps the
    sample-axis layout of the depth gathers and the reduction order along sites.
    """
    depth = np.asarray(depth, dtype=np.float64)

    def paired_columns(lag):
        j = np.searchsorted(positions, positions + lag)
        valid = j < len(positions)
        return np.flatnonzero(valid), j[valid]

    columns = list(map_work(paired_columns, lags))

    def correlate(block):
        rows = len(block)
        out = np.full((rows, len(columns)), np.nan)
        for k, (left, right) in enumerate(columns):
            if left.size:
                a, b = block[:, left], block[:, right]
                a -= a.mean(axis=1, keepdims=True)
                b -= b.mean(axis=1, keepdims=True)
                out[:, k] = (a * b).mean(axis=1) / np.sqrt(
                    (a * a).mean(axis=1) * (b * b).mean(axis=1) + 1e-300
                )
        return out

    # Bound the scratch per item with several rows per block. Evenly split
    # blocks preserve the strided site reductions of a multi-row gather.
    blocks = np.array_split(depth, max(1, (len(depth) + 7) // 8), axis=0)
    return np.concatenate(list(map_work(correlate, blocks)), axis=0)


def _two_means(curves, lags, seed_lag, iterations):
    """Two groups by two-means on the samples' correlation curves, over the lags available for every sample: a lag the
    contig is too short for (no site pairs, a NaN column) is dropped instead of making every distance NaN. Seeded on
    ``seed_lag`` when it is available, else on the lag that varies most."""
    keep = np.all(np.isfinite(curves), axis=0)
    if not keep.any():
        return np.zeros(len(curves), dtype=np.int64)
    seed_index = list(lags).index(seed_lag)
    seed = (int(np.flatnonzero(keep).tolist().index(seed_index)) if keep[seed_index]
            else int(np.argmax(curves[:, keep].var(axis=0))))
    curves = curves[:, keep]
    centre = curves[[np.argmin(curves[:, seed]), np.argmax(curves[:, seed])]]
    group = np.zeros(len(curves), dtype=np.int64)
    for _ in range(iterations):
        group = np.argmin(((curves[:, None, :] - centre[None]) ** 2).sum(axis=2), axis=1)
        if len(np.unique(group)) < 2:
            return group
        centre = np.array([curves[group == k].mean(axis=0) for k in (0, 1)])
    return group


def library_groups(depth, positions, contig: str, parameters: ReadStateParameters, estimator: Callable, z: float,
                   *, map_work=map):
    """(library int64[N], survivals list[ReadSurvival | None], log, strengths): samples grouped into sequencing
    libraries by their read sharing. ``estimator(depth, positions, contig)`` is the evidence stage's read-sharing
    estimator (science.evidence.read_sharing.read_sharing). Divisive: a candidate split of a group by two-means on
    the samples' depth-correlation curves is accepted only if both parts have library_minimum_samples samples, their
    read-sharing
    estimates differ beyond noise (beyond ``z``, the run's family-wise z*, at some lag bin) AND their fitted survival
    laws (each detected at ``z``) differ
    materially (max over library_gaps_bp > library_survival_tolerance). Significance alone splits one library by depth
    with this much data; one group is the outcome when the curves agree. With several libraries each library's
    survival is fitted on its own samples; with one the list holds None (the caller's all-sample fit applies). ``log``
    records every candidate split (level, sizes, contrast z, survival difference, accepted, each part's largest
    excess / SE and fit status); ``strengths`` each library's bin_summary of its own estimate (None for a single
    library: the caller's record)."""
    depth = np.asarray(depth, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    threshold = float(z)
    lags = tuple(int(lag) for lag in parameters.library_lags_bp)
    gaps = np.asarray(parameters.library_gaps_bp, dtype=np.float64)
    log = []

    def divide(members, level, estimate=None, fitted=None):
        leaf = (members, estimate, fitted)
        if level >= parameters.library_levels or len(members) < 2 * parameters.library_minimum_samples:
            return [leaf]
        g = _two_means(_correlation_curves(depth[members], positions, lags, map_work=map_work), lags,
                       int(parameters.library_seed_lag_bp), int(parameters.library_iterations))
        a, b = members[g == 0], members[g == 1]
        if min(len(a), len(b)) < parameters.library_minimum_samples:
            return [leaf]
        sa, sb = estimator(depth[a], positions, contig), estimator(depth[b], positions, contig)
        ea, eb = np.asarray(sa.excess, float), np.asarray(sb.excess, float)
        va, vb = np.asarray(sa.standard_error, float), np.asarray(sb.standard_error, float)
        ok = np.isfinite(ea) & np.isfinite(eb) & np.isfinite(va) & np.isfinite(vb)
        split = float(np.max(np.abs(ea - eb)[ok] / np.sqrt(va[ok] ** 2 + vb[ok] ** 2 + 1e-300))) if ok.any() else 0.0
        fa, fb = fit_survival(sa, parameters, threshold), fit_survival(sb, parameters, threshold)
        difference = float(np.max(np.abs(fa.survival(gaps) - fb.survival(gaps))))
        accepted = split > threshold and difference > parameters.library_survival_tolerance
        log.append(dict(level=level, sizes=(int(len(a)), int(len(b))), z=round(split, 4),
                        survival_difference=round(difference, 4), accepted=bool(accepted),
                        parts_max_z=(round(strongest_bin(sa), 4), round(strongest_bin(sb), 4)),
                        parts_status=(fa.status, fb.status)))
        if not accepted:
            return [leaf]
        # A terminal group's fit was already computed by the split test.
        return divide(a, level + 1, sa, fa) + divide(b, level + 1, sb, fb)

    groups = divide(np.arange(depth.shape[0]), 0)
    library = np.empty(depth.shape[0], dtype=np.int64)
    survivals, strengths = [], []
    for k, (members, estimate, fitted) in enumerate(groups):
        library[members] = k
        if len(groups) > 1:
            survivals.append(fitted)
            strengths.append(bin_summary(estimate))
        else:
            survivals.append(None)
            strengths.append(None)
    return library, survivals, log, strengths
