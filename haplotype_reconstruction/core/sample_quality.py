"""Per-sample, per-contig library-contamination QC from private minor alleles.

Foreign DNA in a library (contamination at read fraction alpha < 1/2) shows
alleles that no other sample carries, on every contig, at a minor-read
fraction near alpha (foreign homozygote) or alpha/2 (foreign heterozygote).
A sample's own heterozygous allele that no other sample carries (an
inherited haplotype segment absent from the sequenced founders, or a founder
segment no sequenced descendant received) shows the heterozygote balance,
about one half, and is regional. Only the first is flagged; the second is
reported.

At the biallelic SNPs that pass the site filter's quality rules r0-r4
(core/site_quality.py, whole analysed cohort), the cohort-minor allele is ALT
when the cohort's ALT reads do not exceed its REF reads, and a sample's cell
carries it with at least one minor read. Per sample on one contig:

* private cells: the sample is the only carrier. Cells of one sample closer
  than fragment_bp can come from the same sequenced fragments, so they form
  one event (each cell weighs 1 / event size);
* the minor reads k of depth d of each private cell follow a mixture of a
  heterozygous allele (fraction h, the contig's r4 clean heterozygote ALT
  fraction, or 1 - h when the minor allele is REF) and a low-fraction allele
  (fraction a, fitted per sample in (0, h/2]: h/2 is the largest fraction of
  a foreign heterozygous allele while foreign DNA is less than half the
  library; errors and weaker contamination lie below it), binomials
  truncated at k >= 1. The low-fraction events are the event-weighted
  posterior mass of the second component at the sample's maximum-likelihood
  fit, so heterozygous cells, however many, are not counted;
* statistic: sqrt(low-fraction events / observed cells), the square root
  being the variance-stabilising scale of event counts. A sample is flagged
  when its robust z against the cohort on that contig (median and
  1.4826 x MAD over samples with observed cells, the spread no smaller than
  the statistic's counting noise) exceeds flag_robust_z. No labels,
  pedigree, benchmark or absolute rate enter.

The decision is per contig, so a contig's result never depends on which other
contigs are run and per-contig resume is unchanged. Foreign DNA that another
analysed sample also carries is not private and is missed: contamination by a
sequenced member of the cohort is not detected.

Inherited-segment diagnostic (never a flag): near-private cells (the sample
and at most one partner carry the allele) whose reads favour the
heterozygous fraction over h/2 at even odds, grouped per (sample, partner)
into fragment events and into regions of events less than segment_gap_bp
apart; regions with at least segment_minimum_events such events are reported
for unflagged samples (a flagged library's fraction-1/2 alleles include its
contaminant's homozygous ones), with the cell count and minor-read fractions
of all the pair's near-private cells in the region.

On the contig where it is flagged, a sample's reads are excluded from founder
construction: block discovery sees it without reads, and feedback, assembly
and founder refinement see its cells as unobserved (uniform and unobserved:
the state of a zero-depth cell), as does the post-painting founder-allele
refit. The site filter's r5 counts carriers among unflagged samples only.
Read calibration, painting, pedigree, family phase and recombination use
every sample's full evidence. The assembly release identity hashes the
founder evidence, so stages that check raw evidence against it apply the
flags block discovery recorded (``founder_evidence_excluded``) for that
check only.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import os
from pathlib import Path

import numpy as np
from scipy.special import expit

from .environment import boolean_setting


VERSION = "private-low-fraction-qc-per-contig-v2"
_MAD_TO_SD = 1.4826
# Numerical settings of the mixture fit, not scientific thresholds.
_BISECTIONS = 50                 # interval halvings (precision 2**-50)
_FIT_ITERATIONS = 200
_FRACTION_TOLERANCE = 1e-7
_MINIMUM_LOW_FRACTION = 1e-6


def configured(default):
    """Route default, overridden by HAPLOTYPES_SAMPLE_QUALITY_FILTER (CLI/TOML)."""
    value = os.environ.get("HAPLOTYPES_SAMPLE_QUALITY_FILTER")
    if value is None:
        return default
    return replace(default, enabled=boolean_setting(value, "sample_quality_filter"))


def identity(config, site_config):
    """Discovery-identity component; None when the QC does not run (it, or the
    site filter whose quality-passing sites it uses, is off), so such
    identities are unchanged."""
    return dict(version=VERSION, config=asdict(config)) if config.enabled and site_config.enabled else None


def _truncated_binomial(k, d, p):
    """log P(k | d, p, k >= 1) without the binomial coefficient (common to both components)."""
    return k * np.log(p) + (d - k) * np.log1p(-p) - np.log1p(-(1.0 - p) ** d)


def _low_log_ratio(k, d, het):
    """Per cell: log likelihood of the low fraction het/2 over the heterozygous fraction het."""
    return _truncated_binomial(k, d, 0.5 * het) - _truncated_binomial(k, d, het)


def _mixing_posterior(log_ratio, weight):
    """Low-component posteriors at the maximum-likelihood mixing weight for fixed components.

    The event-weighted log likelihood sum(w log(pi L_low + (1 - pi) L_het)) is
    concave in pi, so the sign of its slope brackets the maximum; cells that
    favour neither component do not move it.
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

    Mixture of the heterozygous fraction het (fixed) and one low fraction a in
    (0, min(het)/2], both truncated at k >= 1, fitted by coordinate ascent:
    the mixing weight exactly for the current a, then a from the low
    posteriors (the truncated mean d a / (1 - (1 - a)^d) is increasing in a).
    A free a matches the sample's own low cells (single-read errors near 0,
    contamination near alpha), so heterozygous cells are not drawn into a
    misfitted low component.
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


def _robust_z(values, observed, config):
    """Robust z over samples with observed cells; the spread is the cohort's MAD
    but not less than the counting noise of the statistic (a Poisson count's
    square root has standard deviation 1/2, so sqrt(events / cells) has
    0.5 / sqrt(cells)), which keeps the scale defined when most libraries have
    no low-fraction private event. Returns z, median, spread, floor, reason."""
    finite = np.isfinite(values)
    z = np.full(len(values), np.nan)
    if int(finite.sum()) < config.minimum_samples:
        return z, None, None, None, f"fewer than {config.minimum_samples} samples with observed cells"
    median = float(np.median(values[finite]))
    floor = 0.5 / float(np.sqrt(np.median(observed[finite])))
    spread = max(_MAD_TO_SD * float(np.median(np.abs(values[finite] - median))), floor)
    z[finite] = (values[finite] - median) / spread
    return z, median, spread, floor, None


def _listed(values):
    """JSON-compatible list (NaN -> None)."""
    return [None if not np.isfinite(value) else float(value) for value in np.asarray(values, float)]


def assess(alt, dep, positions, heterozygote_balance, config):
    """One contig's per-sample statistics, flags and inherited-segment regions.

    alt, dep: (sites, samples) ALT and total reads of the analysed cohort at
    the contig's r0-r4-passing biallelic SNPs; positions: their coordinates;
    heterozygote_balance: the r4 clean heterozygote ALT fraction.
    """
    alt = np.asarray(alt, np.int64)
    dep = np.asarray(dep, np.int64)
    positions = np.asarray(positions, np.int64)
    n_samples = dep.shape[1]
    minor_is_alt = alt.sum(axis=1) <= (dep - alt).sum(axis=1)
    minor = np.where(minor_is_alt[:, None], alt, dep - alt)
    carrier = minor > 0
    carriers = carrier.sum(axis=1)
    observed = (dep > 0).sum(axis=0)
    # Near-private cells (row-major: by site, then sample); the partner is
    # the other carrier at a two-carrier site, -1 at a private one.
    site, sample = np.nonzero(carrier & (carriers <= 2)[:, None])
    partner = np.full(len(site), -1, np.int64)
    first = np.flatnonzero((carriers[site] == 2) & np.r_[True, site[1:] != site[:-1]])
    partner[first] = sample[first + 1]
    partner[first + 1] = sample[first]
    k = minor[site, sample].astype(float)
    d = dep[site, sample].astype(float)
    het = np.where(minor_is_alt[site], heterozygote_balance, 1.0 - heterozygote_balance)
    position = positions[site]

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

    # Inherited-segment diagnostic: near-private cells whose reads favour the
    # heterozygous fraction over the low one (h/2) at even odds.
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
            partner_fraction = float(minor[site[cells], p].sum() / dep[site[cells], p].sum())
        segments.append(dict(sample=s, partner=p, start=int(first_position), end=int(last_position),
                             events=events, cells=int(len(cells)),
                             minor_read_fraction=float(k[cells].sum() / d[cells].sum()),
                             partner_minor_read_fraction=partner_fraction))
    record = dict(version=VERSION, applied=reason is None, config=asdict(config),
                  passing_sites=int(len(positions)), heterozygote_balance=float(heterozygote_balance),
                  observed_cells=observed.tolist(),
                  near_private_cells=np.bincount(sample, minlength=n_samples).tolist(),
                  private_cells=private_cells.tolist(), private_events=_listed(private_events),
                  low_fraction_events=_listed(low_events), fitted_low_fraction=_listed(low_fraction),
                  private_minor_read_fraction=_listed(private_fraction),
                  heterozygous_like_events=_listed(heterozygous_events),
                  statistic=_listed(statistic), robust_z=_listed(z), flagged=flagged.tolist(),
                  cohort_median=median, cohort_robust_sd=spread, counting_noise_sd=floor, segments=segments)
    if reason is not None:
        record["reason"] = reason
    return record


def _write_table(path, lines):
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)


def run_record_summary(site_record, sample_ids):
    """Run-record product of one contig's decision, or None when the QC did not run."""
    record = site_record.get("sample_quality")
    if record is None:
        return None
    sample_ids = [str(value) for value in sample_ids]
    flagged = np.asarray(record["flagged"], bool)
    names = [sample_ids[index] for index in np.flatnonzero(flagged)]
    return dict(record, sample_ids=sample_ids, flagged_samples=names)


def contig_flags(site_record, *, contig, sample_ids, output_dir):
    """Report one contig's decision (log, TSVs); return the flagged mask.

    The coordinator records run_record_summary(site_record, sample_ids).

    site_record is filter_loaded_sites' record. Returns a boolean over the
    analysed samples, or None when the QC did not run on this contig (QC off,
    site filter off or not applied), so an off run keeps its payloads.
    """
    record = site_record.get("sample_quality")
    if record is None:
        return None
    sample_ids = [str(value) for value in sample_ids]
    flagged = np.asarray(record["flagged"], bool)
    if flagged.shape != (len(sample_ids),):
        raise ValueError("sample-quality flags do not match the analysed samples")
    z = np.array([np.nan if value is None else value for value in record["robust_z"]], float)
    low = np.array([np.nan if value is None else value for value in record["low_fraction_events"]], float)
    names = [sample_ids[index] for index in np.flatnonzero(flagged)]
    order = np.argsort(-np.where(np.isfinite(z), z, -np.inf))[:5]
    segments = record["segments"]
    print(f"    [Sample quality] {contig}: contamination flagged {len(names)} (z > {record['config']['flag_robust_z']})"
          + (f": {', '.join(names)}" if names else "")
          + ("" if record["applied"] else f" (not applied: {record['reason']})")
          + "; highest " + ", ".join(f"{sample_ids[index]} z {z[index]:.1f} ({low[index]:.0f} low-fraction events)"
                                     for index in order)
          + f"; {len(segments)} inherited-segment regions reported", flush=True)
    directory = Path(output_dir) / "sample_quality"
    directory.mkdir(parents=True, exist_ok=True)

    def cell(value, digits):
        return "NA" if value is None else f"{value:.{digits}f}"
    lines = ["sample\tobserved_cells\tnear_private_cells\tprivate_cells\tprivate_events\tlow_fraction_events"
             "\tfitted_low_fraction\tprivate_minor_read_fraction\theterozygous_like_events\tstatistic\trobust_z"
             "\tflagged"]
    for index, sample in enumerate(sample_ids):
        lines.append("\t".join((
            sample, str(record["observed_cells"][index]), str(record["near_private_cells"][index]),
            str(record["private_cells"][index]), cell(record["private_events"][index], 1),
            cell(record["low_fraction_events"][index], 2), cell(record["fitted_low_fraction"][index], 4),
            cell(record["private_minor_read_fraction"][index], 3),
            cell(record["heterozygous_like_events"][index], 1), cell(record["statistic"][index], 5),
            cell(record["robust_z"][index], 3), str(int(flagged[index])))))
    _write_table(directory / f"{contig}.tsv", lines)
    lines = ["sample\tpartner\tstart\tend\tevents\tcells\tminor_read_fraction\tpartner_minor_read_fraction"]
    for row in segments:
        lines.append("\t".join((
            sample_ids[row["sample"]], "-" if row["partner"] < 0 else sample_ids[row["partner"]],
            str(row["start"]), str(row["end"]), f"{row['events']:.1f}", str(row["cells"]),
            f"{row['minor_read_fraction']:.3f}", cell(row["partner_minor_read_fraction"], 3))))
    _write_table(directory / f"{contig}.segments.tsv", lines)
    return flagged


def exclude_from_discovery(genomic_data, excluded):
    """Discovery sees excluded samples without reads (unobserved in every block)."""
    if excluded is None or not np.any(excluded):
        return genomic_data
    for index, reads in enumerate(genomic_data.reads):
        if np.size(reads) == 0:        # the loader's empty blocks have no sample axis
            continue
        reads = np.array(reads, copy=True)
        reads[excluded] = 0
        genomic_data.reads[index] = reads
    return genomic_data


def founder_evidence(probabilities, observed, excluded):
    """Founder-construction evidence: excluded samples' cells uniform and unobserved.

    This is the state of a zero-depth cell, which feedback, assembly and
    founder refinement already treat as state-neutral. Painting and later
    stages keep the unmodified arrays.
    """
    if excluded is None:
        return probabilities, observed
    excluded = np.asarray(excluded, bool)
    if excluded.shape != (len(observed),):
        raise ValueError("founder-evidence exclusion does not match the sample axis")
    if not np.any(excluded):
        return probabilities, observed
    probabilities = np.array(probabilities, copy=True)
    probabilities[excluded] = 1.0 / 3.0
    observed = np.array(observed, dtype=np.bool_, copy=True)
    observed[excluded] = False
    return probabilities, observed
