"""Per-sample, per-contig library-contamination QC from private minor alleles.

Foreign DNA in a library (contamination at read fraction alpha < 1/2) shows alleles that no other
sample carries, on every contig, at a minor-read fraction near alpha (foreign homozygote) or
alpha/2 (foreign heterozygote). A sample's own heterozygous allele that no other sample carries
(an inherited haplotype segment absent from the sequenced founders, or a founder segment no
sequenced descendant received) shows the heterozygote balance, about one half, and is regional.
Only the first is flagged; the second is reported.

At the biallelic SNPs that pass the site filter's quality rules r1-r4 (whole analysed cohort),
the cohort-minor allele is ALT when the cohort's ALT reads do not exceed its REF reads, and a
sample's cell carries it with at least one minor read. Per sample on one contig:

* private cells: the sample is the only carrier. Cells of one sample closer than fragment_bp can
  come from the same sequenced fragments, so they form one event (each cell weighs 1 / event
  size);
* the minor reads k of depth d of each private cell follow a mixture of a heterozygous allele
  (fraction h, the contig's r4 clean heterozygote ALT fraction, or 1 - h when the minor allele is
  REF) and a low-fraction allele (fraction a, fitted per sample in (0, h/2]: h/2 is the largest
  fraction of a foreign heterozygous allele while foreign DNA is less than half the library;
  errors and weaker contamination lie below it), binomials truncated at k >= 1. The low-fraction
  events are the event-weighted posterior mass of the second component at the sample's
  maximum-likelihood fit, so heterozygous cells, however many, are not counted;
* statistic: sqrt(low-fraction events / observed cells), the variance-stabilising scale of event
  counts. A sample is flagged when its robust z against the cohort on that contig (median and
  1.4826 x MAD over samples with observed cells, the spread no smaller than the statistic's
  counting noise) exceeds flag_robust_z. No labels, pedigree, benchmark or absolute rate enter.

The decision is per contig, so a contig's result never depends on which other contigs are run.
Foreign DNA that another analysed sample also carries is not private and is missed:
contamination by a sequenced member of the cohort is not detected.

Inherited-segment diagnostic (never a flag): near-private cells (the sample and at most one
partner carry the allele) whose reads favour the heterozygous fraction over h/2 at even odds,
grouped per (sample, partner) into fragment events and into regions of events less than
segment_gap_bp apart; regions with at least segment_minimum_events such events are reported for
unflagged samples, with the cell count and minor-read fractions of all the pair's near-private
cells in the region.

On the contig where it is flagged, a sample's evidence leaves founder construction
(``ContigEvidence.founder_view``) and the r5 carrier count; read calibration, painting, pedigree,
family phase and recombination use every sample's evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.special import expit

from ..maths import ties
from .config import SampleQualityConfig

VERSION = "private-low-fraction-qc-per-contig-v2"
_MAD_TO_SD = 1.4826
# Numerical settings of the mixture fit, not scientific thresholds.
_BISECTIONS = 50                 # interval halvings (precision 2**-50)
_FIT_ITERATIONS = 200
_FRACTION_TOLERANCE = 1e-7
_MINIMUM_LOW_FRACTION = 1e-6
_CHUNK_SITES = 1024


@dataclass(frozen=True)
class InheritedSegment:
    """A region of heterozygous-like near-private events of one (sample, partner) pair.

    sample, partner: sample-axis indices (partner -1: private to the sample). start, end: bp of the
    first and last event cell. events: fragment events; cells: every near-private cell of the pair
    in the region; minor_read_fraction (and the partner's, None when private).
    """

    sample: int
    partner: int
    start: int
    end: int
    events: float
    cells: int
    minor_read_fraction: float
    partner_minor_read_fraction: float | None


@dataclass(frozen=True)
class SampleQualityResult:
    """One contig's contamination QC over the analysed samples (arrays of length N).

    applied: the cohort was large enough (else ``reason``; nothing is flagged). flagged bool[N].
    Per-sample statistics (NaN where undefined): robust_z, statistic, low_fraction_events,
    fitted_low_fraction, private_events, private_minor_read_fraction, heterozygous_like_events;
    counts: observed_cells, near_private_cells, private_cells. Cohort scale: cohort_median,
    cohort_robust_sd, counting_noise_sd (None when not applied).
    """

    applied: bool
    reason: str | None
    passing_sites: int
    heterozygote_balance: float
    flagged: np.ndarray
    robust_z: np.ndarray
    statistic: np.ndarray
    low_fraction_events: np.ndarray
    fitted_low_fraction: np.ndarray
    private_events: np.ndarray
    private_minor_read_fraction: np.ndarray
    heterozygous_like_events: np.ndarray
    observed_cells: np.ndarray
    near_private_cells: np.ndarray
    private_cells: np.ndarray
    cohort_median: float | None
    cohort_robust_sd: float | None
    counting_noise_sd: float | None
    segments: tuple[InheritedSegment, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)

    _ARRAYS = ("flagged", "robust_z", "statistic", "low_fraction_events", "fitted_low_fraction",
               "private_events", "private_minor_read_fraction", "heterozygous_like_events",
               "observed_cells", "near_private_cells", "private_cells")

    def record(self) -> dict:
        record = {name: getattr(self, name) for name in self._ARRAYS}
        record.update(version=VERSION, applied=self.applied, reason=self.reason,
                      passing_sites=self.passing_sites, heterozygote_balance=self.heterozygote_balance,
                      cohort_median=self.cohort_median, cohort_robust_sd=self.cohort_robust_sd,
                      counting_noise_sd=self.counting_noise_sd, config=dict(self.config),
                      segments=[asdict(segment) for segment in self.segments])
        return record

    @classmethod
    def from_record(cls, record: Mapping) -> "SampleQualityResult":
        values = {name: np.asarray(record[name]) for name in cls._ARRAYS}
        return cls(applied=record["applied"], reason=record["reason"],
                   passing_sites=record["passing_sites"],
                   heterozygote_balance=record["heterozygote_balance"],
                   cohort_median=record["cohort_median"], cohort_robust_sd=record["cohort_robust_sd"],
                   counting_noise_sd=record["counting_noise_sd"], config=record["config"],
                   segments=tuple(InheritedSegment(**row) for row in record["segments"]), **values)


def _truncated_binomial(k, d, p):
    """log P(k | d, p, k >= 1) without the binomial coefficient (common to both components)."""
    return k * np.log(p) + (d - k) * np.log1p(-p) - np.log1p(-(1.0 - p) ** d)


def _low_log_ratio(k, d, het):
    """Per cell: log likelihood of the low fraction het/2 over the heterozygous fraction het."""
    return _truncated_binomial(k, d, 0.5 * het) - _truncated_binomial(k, d, het)


def _mixing_posterior(log_ratio, weight):
    """Low-component posteriors at the maximum-likelihood mixing weight for fixed components.

    The event-weighted log likelihood sum(w log(pi L_low + (1 - pi) L_het)) is concave in pi, so
    the sign of its slope brackets the maximum; cells that favour neither component do not move it.
    """
    even = expit(log_ratio)            # posterior under pi = 1/2
    low, high = 0.0, 1.0
    for _ in range(_BISECTIONS):
        pi = 0.5 * (low + high)
        if np.sum(weight * (2 * even - 1) / (pi * even + (1 - pi) * (1 - even))) > 0:
            low = pi
        else:
            high = pi
    pi = 0.5 * (low + high)
    return expit(log_ratio + np.log(pi) - np.log1p(-pi))


def _low_fraction_fit(k, d, het, weight):
    """Event-weighted low-fraction mass and the fitted low fraction of one sample's private cells.

    Mixture of the heterozygous fraction het (fixed) and one low fraction a in (0, min(het)/2],
    both truncated at k >= 1, fitted by coordinate ascent: the mixing weight exactly for the
    current a, then a from the low posteriors (the truncated mean d a / (1 - (1 - a)^d) is
    increasing in a). A free a matches the sample's own low cells (single-read errors near 0,
    contamination near alpha), so heterozygous cells are not drawn into a misfitted component.
    """
    ceiling = 0.5 * float(np.min(het))
    log_het = _truncated_binomial(k, d, het)
    fraction = ceiling
    for _ in range(_FIT_ITERATIONS):
        posterior = _mixing_posterior(_truncated_binomial(k, d, fraction) - log_het, weight)
        mass = weight * posterior
        target = float(np.sum(mass * k))
        low, high = _MINIMUM_LOW_FRACTION, ceiling
        if np.sum(mass * d * high / (1 - (1 - high) ** d)) > target:
            for _ in range(_BISECTIONS):
                middle = 0.5 * (low + high)
                if np.sum(mass * d * middle / (1 - (1 - middle) ** d)) > target:
                    high = middle
                else:
                    low = middle
        updated = 0.5 * (low + high) if high < ceiling else ceiling
        if abs(updated - fraction) < _FRACTION_TOLERANCE:
            fraction = updated
            break
        fraction = updated
    posterior = _mixing_posterior(_truncated_binomial(k, d, fraction) - log_het, weight)
    return float(np.sum(weight * posterior)), fraction


def _event_weights(group, position, fragment_bp):
    """1 / size of each cell's event; cells are sorted by (group, position)."""
    new = np.ones(len(position), bool)
    new[1:] = (group[1:] != group[:-1]) | (np.diff(position) >= fragment_bp)
    event = np.cumsum(new) - 1
    return 1.0 / np.bincount(event)[event]


def _robust_z(values, observed, config: SampleQualityConfig):
    """Robust z over samples with observed cells.

    The spread is the cohort's MAD but not less than the counting noise of the statistic (a
    Poisson count's square root has standard deviation 1/2, so sqrt(events / cells) has
    0.5 / sqrt(cells)), which keeps the scale defined when most libraries have no low-fraction
    private event. Returns z, median, spread, floor, reason.
    """
    finite = np.isfinite(values)
    z = np.full(len(values), np.nan)
    if int(finite.sum()) < config.minimum_samples:
        return z, None, None, None, f"fewer than {config.minimum_samples} samples with observed cells"
    median = float(np.median(values[finite]))
    floor = 0.5 / float(np.sqrt(np.median(observed[finite])))
    spread = max(_MAD_TO_SD * float(np.median(np.abs(values[finite] - median))), floor)
    z[finite] = (values[finite] - median) / spread
    return z, median, spread, floor, None


def _collect_near_private(tiles, n_samples, heterozygote_balance):
    """Collect at most two cells per site, preserving site-then-sample order.

    A tile supplies int64 ALT/depth arrays (sites, samples) and positions. Partner counts are
    retained with each cell so inherited-segment reporting does not need the dense read arrays.
    """
    columns = [[] for _ in range(8)]
    observed = np.zeros(n_samples, np.int64)
    passing_sites = 0
    for alt, dep, positions in tiles:
        passing_sites += len(positions)
        # At equal read totals ALT is minor by convention (REF under convention reversal).
        alt_total, ref_total = alt.sum(axis=1), (dep - alt).sum(axis=1)
        minor_is_alt = ((alt_total < ref_total) if ties.reverse_conventions()
                        else (alt_total <= ref_total))
        minor = np.where(minor_is_alt[:, None], alt, dep - alt)
        carrier = minor > 0
        carriers = carrier.sum(axis=1)
        observed += (dep > 0).sum(axis=0)
        site, sample = np.nonzero(carrier & (carriers <= 2)[:, None])
        partner = np.full(len(site), -1, np.int64)
        first = np.flatnonzero((carriers[site] == 2) & np.r_[True, site[1:] != site[:-1]])
        partner[first] = sample[first + 1]
        partner[first + 1] = sample[first]
        shared = partner >= 0
        partner_k = np.zeros(len(site), np.int64)
        partner_d = np.zeros(len(site), np.int64)
        partner_k[shared] = minor[site[shared], partner[shared]]
        partner_d[shared] = dep[site[shared], partner[shared]]
        values = (
            minor[site, sample].astype(float), dep[site, sample].astype(float),
            np.where(minor_is_alt[site], heterozygote_balance, 1.0 - heterozygote_balance),
            positions[site], sample, partner, partner_k, partner_d,
        )
        for column, value in zip(columns, values):
            column.append(value)
    dtypes = (float, float, float, np.int64, np.int64, np.int64, np.int64, np.int64)
    cells = tuple(np.concatenate(column) if column else np.empty(0, dtype)
                  for column, dtype in zip(columns, dtypes))
    return passing_sites, observed, cells


def assess_counts(counts, positions, passing, heterozygote_balance, config: SampleQualityConfig, *,
                  chunk_sites: int = _CHUNK_SITES) -> SampleQualityResult:
    """Assess raw counts without allocating arrays across the full sample/site axes.

    counts: int32 (samples, input sites, 2), REF then ALT; passing: bool[input sites] selecting
    r1-r4-passing sites. Dense scratch is bounded by ``chunk_sites``; retained cell summaries
    have at most two entries per passing site. The statistical operations match ``assess``.
    """
    def tiles():
        for start in range(0, len(positions), chunk_sites):
            selected = np.flatnonzero(passing[start:start + chunk_sites]) + start
            if not len(selected):
                continue
            alt = np.asarray(counts[:, selected, 1].T, np.int64)
            dep = np.asarray(counts[:, selected, :].sum(axis=2, dtype=np.int32).T, np.int64)
            yield alt, dep, np.asarray(positions[selected], np.int64)

    n_samples = counts.shape[0]
    passing_sites, observed, cells = _collect_near_private(tiles(), n_samples, heterozygote_balance)
    return _assess_cells(n_samples, passing_sites, observed, *cells, heterozygote_balance, config)


def assess(alt, dep, positions, heterozygote_balance, config: SampleQualityConfig) -> SampleQualityResult:
    """One contig's per-sample statistics, flags and inherited-segment regions.

    alt, dep: (sites, samples) ALT and total reads at r1-r4-passing SNPs; positions: their
    coordinates; heterozygote_balance: the r4 clean heterozygote ALT fraction. Conversion and
    minor-allele selection use bounded tiles, including for callers with non-int64 inputs.
    """
    alt, dep, positions = np.asarray(alt), np.asarray(dep), np.asarray(positions)

    def tiles():
        for start in range(0, len(positions), _CHUNK_SITES):
            stop = start + _CHUNK_SITES
            yield (np.asarray(alt[start:stop], np.int64), np.asarray(dep[start:stop], np.int64),
                   np.asarray(positions[start:stop], np.int64))

    n_samples = dep.shape[1]
    passing_sites, observed, cells = _collect_near_private(tiles(), n_samples, heterozygote_balance)
    return _assess_cells(n_samples, passing_sites, observed, *cells, heterozygote_balance, config)


def _assess_cells(n_samples, passing_sites, observed, k, d, het, position, sample, partner,
                  partner_k, partner_d, heterozygote_balance, config):
    """The unchanged event fits and cohort statistics, using only near-private cells."""
    private_cells = np.zeros(n_samples, np.int64)
    private_events = np.zeros(n_samples)
    low_events = np.zeros(n_samples)
    low_fraction = np.full(n_samples, np.nan)
    private_fraction = np.full(n_samples, np.nan)
    private = np.flatnonzero(partner < 0)
    private = private[np.lexsort((position[private], sample[private]))]
    weight = _event_weights(sample[private], position[private], config.fragment_bp)
    first_of_sample = np.ones(len(private), bool)
    first_of_sample[1:] = sample[private][1:] != sample[private][:-1]
    bounds = np.flatnonzero(np.r_[first_of_sample, True])
    for start, stop in zip(bounds[:-1], bounds[1:]):
        cells, w = private[start:stop], weight[start:stop]
        s = sample[cells[0]]
        private_cells[s] = len(cells)
        private_events[s] = w.sum()
        low_events[s], low_fraction[s] = _low_fraction_fit(k[cells], d[cells], het[cells], w)
        private_fraction[s] = k[cells].sum() / d[cells].sum()
    statistic = np.sqrt(np.divide(low_events, observed, out=np.full(n_samples, np.nan),
                                  where=observed > 0))
    z, median, spread, floor, reason = _robust_z(statistic, observed, config)
    flagged = np.zeros(n_samples, bool)
    flagged[np.isfinite(z)] = z[np.isfinite(z)] > config.flag_robust_z

    # Inherited-segment diagnostic: near-private cells whose reads favour the heterozygous
    # fraction over the low one (h/2) at even odds.
    hetlike = np.flatnonzero(_low_log_ratio(k, d, het) <= 0)
    hetlike = hetlike[np.lexsort((position[hetlike], partner[hetlike], sample[hetlike]))]
    pair = sample[hetlike] * (n_samples + 1) + partner[hetlike] + 1
    weight = _event_weights(pair, position[hetlike], config.fragment_bp)
    heterozygous_events = np.bincount(sample[hetlike], weights=weight, minlength=n_samples)
    region = np.ones(len(hetlike), bool)
    region[1:] = (pair[1:] != pair[:-1]) | (np.diff(position[hetlike]) >= config.segment_gap_bp)
    bounds = np.flatnonzero(np.r_[region, True])
    segments = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        events = float(weight[start:stop].sum())
        s, p = int(sample[hetlike[start]]), int(partner[hetlike[start]])
        if events < config.segment_minimum_events or flagged[s]:
            continue
        first_position, last_position = position[hetlike[start]], position[hetlike[stop - 1]]
        # Every near-private cell of the pair in the region, whatever its reads.
        cells = np.flatnonzero((sample == s) & (partner == p) & (position >= first_position)
                               & (position <= last_position))
        partner_fraction = None
        if p >= 0:
            partner_fraction = float(partner_k[cells].sum() / partner_d[cells].sum())
        segments.append(InheritedSegment(
            sample=s, partner=p, start=int(first_position), end=int(last_position), events=events,
            cells=int(len(cells)), minor_read_fraction=float(k[cells].sum() / d[cells].sum()),
            partner_minor_read_fraction=partner_fraction))
    return SampleQualityResult(
        applied=reason is None, reason=reason, passing_sites=passing_sites,
        heterozygote_balance=float(heterozygote_balance), flagged=flagged, robust_z=z,
        statistic=statistic, low_fraction_events=low_events, fitted_low_fraction=low_fraction,
        private_events=private_events, private_minor_read_fraction=private_fraction,
        heterozygous_like_events=heterozygous_events, observed_cells=observed,
        near_private_cells=np.bincount(sample, minlength=n_samples), private_cells=private_cells,
        cohort_median=median, cohort_robust_sd=spread, counting_noise_sd=floor,
        segments=tuple(segments), config=asdict(config))


def identity(config: SampleQualityConfig) -> dict:
    return dict(version=VERSION, config=asdict(config))
