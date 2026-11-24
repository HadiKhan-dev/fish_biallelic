"""Per-library contamination: its read fraction alpha, the sequenced donor it comes from, and contamination-aware
genotype likelihoods (4.22 QC A).

A contaminated library's reads at a site are a mixture of its own genotype g and foreign DNA:

    P(k ALT of d reads | g, alpha) = Bin(k; d, (1 - alpha) theta(g) + alpha q),

theta(g) = (e, h, 1 - e) the library's own read model (evidence: homozygote error e, heterozygote ALT mean h), and q
the foreign ALT probability at the site. Two foreign laws:

- population: q = the other libraries' mean ALT read fraction at the site (leave one out). A plug-in for an unknown
  source; exact for many donors, only partial for one unsequenced donor (QC A2: 9 of 26 lost edges recovered);
- donor: q = phi(g_D) = (e_D, 1/2, 1 - e_D) for the painted genotype g_D of one sequenced library D. A heavily
  contaminated library usually carries one other individual's DNA; when that individual is sequenced the donor law
  is the right one (QC A2: every lost edge recovered), and the population plug-in is harmful (it loses clean edges).

The own genotype g is the library's painted genotype (its founder mosaic). In a cross, genotypes follow the founder
mosaic, not Hardy-Weinberg proportions at the cohort frequency: an HWE own-genotype prior absorbed its misfit as
alpha and hid contamination by relatives (QC section 4). Painted genotypes agree with the truth at ~99.8 % of sites
even in a 30 %-contaminated library, so painting does not absorb the contamination it is measured against.

Estimation uses additive sufficient statistics, so pooling contigs is a sum and each likelihood evaluation is O(cells):
population cells (g, q bin) with q binned at a resolution far below any alpha of interest; donor cells (g, g_D).
alpha is profiled on a fixed grid, then refined by golden-section search inside the best grid interval. Flagging
needs alpha above an absolute floor AND an empirical null over the cohort (the robust z of alpha): a chi-square
threshold on the likelihood ratio is useless here because clean libraries carry a misfit floor of 1e3 nats
(painted-genotype errors, het balance).

Method: docs/methods/painting.md#contamination
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..shared.alleles import complement_calls

# Profile grid of alpha (read fraction): 0, then geometric from 1e-3 to 1/2. 1/2 is a mix-up, the identifiable limit
# (beyond it own and foreign genotypes swap roles).
ALPHA_GRID = np.concatenate(([0.0], np.geomspace(1e-3, 0.5, 48)))
GOLDEN_STEPS = 40
# Own read-error grid profiled jointly with alpha: the evidence read model's per-library error absorbs contamination
# (a 30 %-contaminated library fitted e = 9.4e-3 against 1.5e-5 to 7.7e-5 for clean ones, QC A3), so e is re-estimated
# here. Contamination is separable from error because its minor reads scale with the foreign frequency q and error's
# do not. 1e-5 to 2e-2, 14 points (half-decade steps).
ERROR_GRID = np.geomspace(1e-5, 2e-2, 14)
# Bins of the foreign ALT probability q for the population cells: width 1/256 moves p by at most alpha / 512.
Q_BINS = 256
_PROBABILITY_FLOOR = 1e-12


@dataclass(frozen=True)
class AlphaFit:
    """Profile fit of one library under one foreign law: alpha_hat and its own error e_hat, the log likelihood there
    and the best log likelihood at alpha = 0 (over the same error grid)."""

    alpha: float
    error: float
    log_likelihood: float
    log_likelihood_null: float

    @property
    def statistic(self) -> float:
        """Likelihood-ratio statistic 2 (l(alpha_hat) - l(0)), >= 0."""
        return max(0.0, 2.0 * (self.log_likelihood - self.log_likelihood_null))


def own_probabilities(error: float, heterozygote: float) -> np.ndarray:
    """theta(g), g = 0, 1, 2: the ALT read probability of the library's own reads."""
    return np.array([error, heterozygote, 1.0 - error], dtype=np.float64)


def donor_probabilities(error: float) -> np.ndarray:
    """phi(g_D): the ALT read probability of a donor's reads (its own error, an even heterozygote)."""
    return np.array([error, 0.5, 1.0 - error], dtype=np.float64)


def cohort_sums(alt: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per site: the sum of the libraries' ALT read fractions and the number of libraries with reads (int [N, L])."""
    observed = depth > 0
    fraction = np.divide(alt, depth, out=np.zeros(alt.shape, dtype=np.float64), where=observed)
    return fraction.sum(axis=0), observed.sum(axis=0)


def leave_one_out_frequency(alt_i, depth_i, fraction_sum, observed_count) -> np.ndarray:
    """q at every site for library i: the other libraries' mean ALT read fraction (nan where none has reads)."""
    own = np.divide(alt_i, depth_i, out=np.zeros(len(alt_i), dtype=np.float64), where=depth_i > 0)
    others = observed_count - (depth_i > 0)
    return np.divide(fraction_sum - own, others, out=np.full(len(alt_i), np.nan), where=others > 0)


def population_cells(alt_i, depth_i, genotype_i, q) -> tuple[np.ndarray, np.ndarray]:
    """Sufficient statistics (ALT, REF) [3, Q_BINS] of library i's reads at sites with a painted genotype and a q."""
    use = (genotype_i >= 0) & (depth_i > 0) & np.isfinite(q)
    g = genotype_i[use].astype(np.int64)
    b = np.minimum((q[use] * Q_BINS).astype(np.int64), Q_BINS - 1)
    cell = g * Q_BINS + b
    k = alt_i[use].astype(np.float64)
    d = depth_i[use].astype(np.float64)
    alt_cells = np.bincount(cell, weights=k, minlength=3 * Q_BINS).reshape(3, Q_BINS)
    ref_cells = np.bincount(cell, weights=d - k, minlength=3 * Q_BINS).reshape(3, Q_BINS)
    return alt_cells, ref_cells


def donor_cells(alt_i, depth_i, genotype_i, genotype_d) -> tuple[np.ndarray, np.ndarray]:
    """Sufficient statistics (ALT, REF) [3, 3] over (g_i, g_D) at sites where both are painted."""
    use = (genotype_i >= 0) & (genotype_d >= 0) & (depth_i > 0)
    cell = genotype_i[use].astype(np.int64) * 3 + genotype_d[use].astype(np.int64)
    k = alt_i[use].astype(np.float64)
    d = depth_i[use].astype(np.float64)
    return (np.bincount(cell, weights=k, minlength=9).reshape(3, 3),
            np.bincount(cell, weights=d - k, minlength=9).reshape(3, 3))


def _profile_alpha(log_likelihood) -> tuple[float, float, float]:
    """(alpha_hat, l(alpha_hat), l(0)) of a one-parameter log likelihood: grid, then golden section in the best
    grid interval."""
    values = np.array([log_likelihood(a) for a in ALPHA_GRID])
    best = int(np.argmax(values))
    lo = ALPHA_GRID[max(best - 1, 0)]
    hi = ALPHA_GRID[min(best + 1, len(ALPHA_GRID) - 1)]
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = log_likelihood(a), log_likelihood(b)
    for _ in range(GOLDEN_STEPS):
        if fa >= fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = log_likelihood(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = log_likelihood(b)
    candidates = ((float(ALPHA_GRID[best]), float(values[best])), (a, fa), (b, fb))
    alpha, value = max(candidates, key=lambda item: (item[1], -item[0]))
    return alpha, value, float(values[0])


def _profile(log_likelihood) -> AlphaFit:
    """Joint profile over the own error grid and alpha: log_likelihood(alpha, error)."""
    best = None
    null = -np.inf
    for error in ERROR_GRID:
        alpha, value, at_zero = _profile_alpha(lambda a, e=error: log_likelihood(a, e))
        null = max(null, at_zero)
        if best is None or value > best[2]:
            best = (alpha, float(error), value)
    return AlphaFit(alpha=best[0], error=best[1], log_likelihood=best[2], log_likelihood_null=null)


def fit_population(alt_cells, ref_cells, heterozygote: float) -> AlphaFit:
    """alpha and the own error under the population law from (g, q-bin) cells; q at the bin centre; the heterozygote
    ALT mean h from the evidence read model."""
    centres = (np.arange(Q_BINS) + 0.5) / Q_BINS

    def log_likelihood(alpha: float, error: float) -> float:
        theta = own_probabilities(error, heterozygote)[:, None]
        p = np.clip((1.0 - alpha) * theta + alpha * centres[None, :], _PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)
        return float((alt_cells * np.log(p) + ref_cells * np.log1p(-p)).sum())

    return _profile(log_likelihood)


def fit_donor(alt_cells, ref_cells, heterozygote: float, donor_error: float) -> AlphaFit:
    """alpha and the own error under one donor's law from (g, g_D) cells."""
    phi = donor_probabilities(donor_error)[None, :]

    def log_likelihood(alpha: float, error: float) -> float:
        theta = own_probabilities(error, heterozygote)[:, None]
        p = np.clip((1.0 - alpha) * theta + alpha * phi, _PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)
        return float((alt_cells * np.log(p) + ref_cells * np.log1p(-p)).sum())

    return _profile(log_likelihood)


def same_homozygote_fraction(alt_cells, ref_cells) -> float:
    """Minor-read fraction at sites where the library and the donor are painted homozygous for the same allele, from
    (g_i, g_D) cells: a true donor adds no foreign minor reads there, a stand-in for an unsequenced source does."""
    minor = alt_cells[0, 0] + ref_cells[2, 2]
    total = alt_cells[0, 0] + ref_cells[0, 0] + alt_cells[2, 2] + ref_cells[2, 2]
    return float(minor / total) if total > 0 else float("nan")


def donor_statistics(alt_cells, ref_cells, heterozygote: float, donor_errors) -> np.ndarray:
    """The donor-law likelihood-ratio statistic of every candidate at once (the scan's ranking), on the alpha and own
    error grids without refinement: alt_cells, ref_cells [J, 3, 3]; donor_errors [J]. The best candidate is then
    refined by fit_donor. Vectorised over candidates: the scan is O(J) array work, not J Python fits."""
    alt_cells = np.asarray(alt_cells, dtype=np.float64)
    ref_cells = np.asarray(ref_cells, dtype=np.float64)
    errors = np.asarray(donor_errors, dtype=np.float64)
    phi = np.stack((errors, np.full_like(errors, 0.5), 1.0 - errors), axis=1)  # [J, 3]
    theta = np.stack((ERROR_GRID, np.full_like(ERROR_GRID, heterozygote), 1.0 - ERROR_GRID), axis=1)  # [E, 3]
    best = np.full(len(errors), -np.inf)
    null = np.full(len(errors), -np.inf)
    for a in ALPHA_GRID:
        p = (1.0 - a) * theta[None, :, :, None] + a * phi[:, None, None, :]  # [J, E, g, gD]
        p = np.clip(p, _PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)
        value = (alt_cells[:, None] * np.log(p) + ref_cells[:, None] * np.log1p(-p)).sum(axis=(2, 3))  # [J, E]
        top = value.max(axis=1)
        best = np.maximum(best, top)
        if a == 0.0:
            null = top
    return np.maximum(0.0, 2.0 * (best - null))


def robust_z(statistics: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 MAD) over the cohort; MAD floored at the smallest positive deviation so a near-constant
    cohort does not divide by zero."""
    x = np.asarray(statistics, dtype=np.float64)
    centre = np.median(x)
    deviation = np.abs(x - centre)
    mad = np.median(deviation)
    if not mad > 0:
        positive = deviation[deviation > 0]
        mad = float(positive.min()) if positive.size else 1.0
    return (x - centre) / (1.4826 * mad)


def donor_frequency(genotype_d, donor_error: float, donor_likelihoods, q) -> np.ndarray:
    """The donor law's q at every site: phi(painted g_D); where the donor is unpainted, phi averaged over its genotype
    posterior (its GL row with a Hardy-Weinberg prior at the cohort frequency q, the fallback of the A2 prototype)."""
    phi = donor_probabilities(donor_error)
    painted = genotype_d >= 0
    out = np.empty(len(genotype_d), dtype=np.float64)
    out[painted] = phi[genotype_d[painted].astype(np.int64)]
    if (~painted).any():
        qq = np.nan_to_num(q[~painted], nan=0.5)
        prior = np.stack(((1.0 - qq) ** 2, 2.0 * qq * (1.0 - qq), qq**2), axis=1)
        posterior = np.asarray(donor_likelihoods, dtype=np.float64)[~painted] * prior
        posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny)
        out[~painted] = posterior @ phi
    return out


def corrected_likelihoods(alt_i, depth_i, error: float, heterozygote: float, alpha: float, q) -> np.ndarray:
    """Normalised genotype likelihoods [L, 3] of library i under the contamination model (binomial own reads; the
    evidence read model's heterozygote overdispersion is not carried, a stated simplification for corrected libraries
    only); 1/3 each at sites without reads; q missing (no other library observed) = 1/2."""
    k = np.asarray(alt_i, dtype=np.float64)[:, None]
    d = np.asarray(depth_i, dtype=np.float64)[:, None]
    theta = own_probabilities(error, heterozygote)[None, :]
    qq = np.nan_to_num(np.asarray(q, dtype=np.float64), nan=0.5)[:, None]
    p = np.clip((1.0 - alpha) * theta + alpha * qq, _PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)
    log_value = k * np.log(p) + (d - k) * np.log1p(-p)
    log_value -= log_value.max(axis=1, keepdims=True)
    value = np.exp(log_value)
    value /= value.sum(axis=1, keepdims=True)
    value[d[:, 0] == 0] = 1.0 / 3.0
    return value


def painted_genotypes(
    painting, positions: np.ndarray, rows=None, *, search_alleles_reversed: bool = False
) -> np.ndarray:
    """int8 [N, L] (or [len(rows), L] for the given sample rows) painted genotype
    (ALT dosage of the two released classes) of every sample at the given evidence
    positions in the requested coding; -1 where a track abstains, a class is uncalled at the site, or the site is not
    a painting site. A class
    value below C names a class (a local key is its group's smallest class, whose alleles are the group's over the
    bin); C (unknown) and negative values abstain."""
    positions = np.asarray(positions, dtype=np.int64)
    selected = list(range(len(painting.sample_ids))) if rows is None else [int(r) for r in rows]
    out = np.full((len(selected), len(positions)), -1, dtype=np.int8)
    for component in painting.components:
        bins, states = component.bins, component.states
        if bins.n_bins == 0:
            continue
        site_positions = np.asarray(states.positions, dtype=np.int64)
        columns = np.searchsorted(positions, site_positions)
        hit = (columns < len(positions)) & (positions[np.minimum(columns, len(positions) - 1)] == site_positions)
        named = np.asarray(states.named_alleles)
        if search_alleles_reversed:
            named = complement_calls(named)
        n_classes = named.shape[0]
        site_bin = np.repeat(np.arange(bins.n_bins), np.asarray(bins.stops) - np.asarray(bins.starts))
        sites = np.arange(len(site_bin))
        classes = np.asarray(component.classes)  # [N, 2, B]
        for slot, row in enumerate(selected):
            first, second = classes[row, 0, site_bin], classes[row, 1, site_bin]
            valid = (first >= 0) & (second >= 0) & (first < n_classes) & (second < n_classes)
            a = np.where(valid, named[np.where(valid, first, 0), sites], -1)
            b = np.where(valid, named[np.where(valid, second, 0), sites], -1)
            genotype = np.where((a >= 0) & (b >= 0), a + b, -1).astype(np.int8)
            out[slot, columns[hit]] = genotype[hit]
    return out


def genotype_indicators(genotypes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """float64 [N, L] indicators of painted genotype 0, 1, 2 (built once per contig for every donor block)."""
    return tuple((genotypes == g).astype(np.float64) for g in range(3))


def donor_cell_rows(alt, depth, genotypes, lo: int, hi: int, indicators=None) -> tuple[np.ndarray, np.ndarray]:
    """Donor sufficient statistics of libraries lo..hi-1 against every donor D: ALT and REF read sums
    [hi - lo, N, 3, 3] over sites where both are painted, (g_i, g_D) cells; nine indicator products (BLAS).
    ``indicators``: genotype_indicators(genotypes), shared by every block (rebuilding them per block multiplied the
    memory by the concurrent blocks)."""
    donors = genotype_indicators(genotypes) if indicators is None else indicators
    rows_alt = np.asarray(alt[lo:hi], dtype=np.float64)
    rows_ref = np.asarray(depth[lo:hi], dtype=np.float64) - rows_alt
    alt_cells = np.empty((hi - lo, genotypes.shape[0], 3, 3))
    ref_cells = np.empty_like(alt_cells)
    for g in range(3):
        mask = donors[g][lo:hi]
        weighted_alt, weighted_ref = rows_alt * mask, rows_ref * mask
        for h in range(3):
            alt_cells[:, :, g, h] = weighted_alt @ donors[h].T
            ref_cells[:, :, g, h] = weighted_ref @ donors[h].T
    return alt_cells, ref_cells


def all_donor_cells(alt, depth, genotypes) -> tuple[np.ndarray, np.ndarray]:
    """Donor sufficient statistics of every ordered pair (i, D): [N, N, 3, 3] ALT and REF read sums."""
    return donor_cell_rows(alt, depth, genotypes, 0, genotypes.shape[0])


@dataclass(frozen=True)
class LibraryContamination:
    """Run-wide decision for every library (axis N): alpha (population law) and its own error, the robust z of alpha,
    flagged (corrected downstream), donor (sample index, -1 = no sequenced donor named) with its alpha, error and
    statistic, and the best candidate's same-homozygote residual (nan when not scanned)."""

    alpha: np.ndarray
    error: np.ndarray
    statistic: np.ndarray
    z: np.ndarray
    flagged: np.ndarray
    donor: np.ndarray
    donor_alpha: np.ndarray
    donor_error: np.ndarray
    donor_statistic: np.ndarray
    residual: np.ndarray


def decide(population, donors, *, alpha_floor: float, z_floor: float, donor_residual: float,
           baseline: float) -> LibraryContamination:
    """Flag a library when alpha is at least alpha_floor above the cohort's median alpha (its clean level: the misfit
    floor that clean libraries share, which rises with a noisier cohort) and the robust z of alpha >= z_floor. Name its
    best donor (the
    largest donor-law likelihood-ratio statistic on their shared sites) when the same-homozygote residual
    (same_homozygote_fraction - baseline) / alpha_D is at most ``donor_residual``: a true donor adds no foreign minor
    reads where both are the same homozygote, so the residual stays near 0, while a stand-in leaves foreign reads there
    that grow with alpha. ``baseline``: the unflagged libraries' same-homozygote minor fraction (their own error).

    population: AlphaFit per library; donors: per library None (not scanned) or (donor index, AlphaFit,
    same-homozygote fraction)."""
    n = len(population)
    alpha = np.array([f.alpha for f in population])
    z = robust_z(alpha)
    flagged = (alpha - np.median(alpha) >= alpha_floor) & (z >= z_floor)
    donor = np.full(n, -1, dtype=np.int64)
    donor_alpha = np.zeros(n)
    donor_error = np.zeros(n)
    donor_statistic = np.zeros(n)
    residual = np.full(n, np.nan)
    for i in np.flatnonzero(flagged):
        if donors[i] is None:
            continue
        j, fit, same = donors[i]
        donor_statistic[i] = fit.statistic
        if fit.alpha > 0 and np.isfinite(same):
            residual[i] = (same - baseline) / fit.alpha
            if residual[i] <= donor_residual:
                donor[i], donor_alpha[i], donor_error[i] = j, fit.alpha, fit.error
    return LibraryContamination(
        alpha=alpha, error=np.array([f.error for f in population]),
        statistic=np.array([f.statistic for f in population]), z=z, flagged=flagged, donor=donor,
        donor_alpha=donor_alpha, donor_error=donor_error, donor_statistic=donor_statistic, residual=residual,
    )

__all__ = [
    "ALPHA_GRID",
    "AlphaFit",
    "ERROR_GRID",
    "LibraryContamination",
    "all_donor_cells",
    "decide",
    "donor_cell_rows",
    "donor_statistics",
    "genotype_indicators",
    "painted_genotypes",
    "cohort_sums",
    "corrected_likelihoods",
    "donor_cells",
    "donor_frequency",
    "donor_probabilities",
    "fit_donor",
    "fit_population",
    "leave_one_out_frequency",
    "own_probabilities",
    "population_cells",
    "robust_z",
    "same_homozygote_fraction",
]
