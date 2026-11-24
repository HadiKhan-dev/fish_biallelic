"""Sequencing reads of simulated individuals (the generating read model, never the inference one).

Two samplers, chosen by ``ReadModel.read_length_bp``:

* independent sites (0): per cell, depth ~ Poisson(depth x sample factor / (1 - dropout share))
  and ALT reads ~ Binomial(depth, p_g), p = (error_rate, heterozygote_alt_probability,
  1 - error_rate); a lognormal sample factor (mean one, coefficient of variation depth_cv) and one
  contiguous dropout tract of dropout_fraction of the sites per sample are drawn only when
  configured. heterozygote_alt_probability exists only here: this sampler has no reads, so a
  per-cell balance is its whole law;
* paired-end fragments (> 0): per sample, fragment starts form a Poisson process along the contig;
  a fragment has insert length F ~ Normal(fragment_mean_bp, fragment_sd_bp) rounded and clipped to
  [50 bp, mean + 6 sd], comes from one of the sample's two haplotypes with probability 1/2 and is
  read over read_length_bp at each end (bases of overlapping mates counted once). The start rate is
  depth / E[bases covered per fragment], so every site keeps the Poisson(depth) marginal while
  neighbouring sites become dependent. depth_cv and dropout_fraction act as above. Allelic imbalance
  is read-level (reference mapping bias): each mate fails to align, losing every base it covers, with
  probability 1 - exp(-reference_bias k**reference_bias_power), k the ALT alleles of the fragment's
  homologue at the sites under that mate's own span, so reads of a haplotype that differs densely from
  the reference lose most. A lost first mate leaves the overlap to the second. A kept read therefore
  shows its homologue's alleles at every site it covers; the error rate acts on homozygous cells only.

Both short-read samplers independently leave each covered base uncalled with probability
uncalled_base_probability. Configured depth is before this loss. Binomial thinning of the
REF/ALT counts is equivalent to tossing a separate coin for each hit, including hits sharing
one fragment. The loss is drawn at the end of each sample's stream, after all other draws.

Optional site factors multiply shared mean-one lognormal draws by a uniform retained
share in a Bernoulli-impaired fraction of sites/windows. Loss is not renormalised:
depth means unimpaired depth, with expected sample mean
depth * (1 - site_loss_fraction / 2) * (1 - site_loss_fine_fraction / 2).
Independent fine-window loss multiplies the existing shared site factor, using the
same uniform-retained-share law and draws after the coarse lognormal and C1 loss.
Regional factors are independent mean-one lognormal draws per sample/physical window,
from the sample's own regional stream; the reads are then sampled once at those intensities.
With regional CV zero that stream is not created and no count changes. Optional independent gamma cell factors have mean one
and variance count_dispersion. Conditional depth is Poisson in both samplers, hence gamma
mixing gives negative-binomial marginals, Var(D | sample, site) = mu + alpha * mu**2.
Fragments are drawn at the maximum target factor per sample and site-hits thinned to their
target intensity, retaining the shared fragment's homologue. No extra draws or arithmetic
on the unmodified rates occur when these options are off.

* long reads (long_read_mean_bp > 0): per sample, read starts form a Poisson process along the contig; a read has
  length ~ Lognormal (mean long_read_mean_bp, log-SD long_read_length_sigma) rounded and clipped to
  [LONG_READ_MIN_BP, exp(log-mean + LONG_READ_SIGMA_LIMIT x SD)], comes from one homologue with probability 1/2 and
  covers every site it spans; depth_cv, dropout_fraction, site and regional factors and count_dispersion act as for
  fragments. Each read draws its own error rate e = 10**(-Q/10), Q ~ Normal(read_quality_mean, read_quality_sd)
  clipped to [READ_QUALITY_MIN, READ_QUALITY_MAX], used at every site it covers (errors correlated within a read).
  At a covered site a read's base is: lost to a deletion error (e (1 - substitution_share) / 2), to a substitution
  into a third base (2 e substitution_share / 3) or to reference bias (1 - exp(-reference_bias k**power), k = the
  read homologue's ALT alleles within reference_bias_window_bp, the site included);
  read as the other allele (substitution, e substitution_share / 3; an ALT base also as REF with probability
  reference_pull x e / mean e); or kept. At a context site (a contig-level share context_sites, all samples alike), the
  allele that extends a homopolymer (drawn per site) has an extra indel probability
  min(1, e (1 - substitution_share) context_factor s_i), s_i a mean-one exponential strength of the site (run length
  and sequence vary between sites), of which context_convert is read as the other allele and the rest lost. The causes
  combine as independent hazards. Cross-sample reads: below.
  Context sites come from the contig stream after every draw of the other samplers, so those are unchanged.

Cross-sample reads. Uniform foreign reads (foreign_read_share > 0, long reads and paired-end fragments): each read or
fragment comes, with that probability, from a uniform other sequenced sample's homologue (the read's own homologue
choice), whole (both mates), with the sampler's own geometry, depth thinning and mapping loss (k counted on the donor
homologue's alleles); errors then follow the donor's genotype. Per-library contamination (contamination_share law or
an explicit contamination_samples list; every sampler): a contaminated library takes each read from its one donor
with probability its share (a single donor per library, as heavy contamination or a mix-up usually is; the law draws
the donor uniformly among the other sequenced samples, genome-wide, from simulation.contamination_level keyed by the
generating seed and the sample identifier). Both draw from the read batch's stream after the homologue choice, uniform
foreign reads first (exactly the long-read sampler's earlier draws), and only when their share is positive; the
independent-site sampler replaces each read by a donor read with probability the share after its other draws.

Exogenous (other-species) reads, any sampler (exogenous_read_share > 0): a share exogenous_window_fraction of the
occupied windows of exogenous_window_bp (shared by all samples, drawn from the contig stream after every other draw)
is conserved enough for another species' reads to map; each site there carries that species' base, ALT with
probability exogenous_alt_probability, fixed per site (one foreign genome), REF otherwise. A sample's exogenous reads
are the share s of its mapped reads (s from simulation.exogenous_level, genome-wide per sample: contaminated with
probability exogenous_sample_fraction at exogenous_read_share x Exponential(1)), all placed inside the conserved
windows: depth there rises by depth x factor x s / (1 - s) x (sites' span / conserved bp). Reads follow the sampler's
geometry (paired mates, long reads, or independent sites), are clipped to their window (a read running into
non-conserved sequence fails there) and carry no sequencing error. They are drawn from the sample's own
simulation.exogenous_reads stream after the sampler and structure draws; nothing is drawn when the share is zero.

PCR duplicates (paired-end, duplicate_rate > 0): a library's rate d is duplicate_rate x Exponential(1) (clipped at
DUPLICATE_RATE_LIMIT; stream simulation.duplicate_level keyed by the generating seed and the sample identifier). Each
sequenced fragment has Poisson(d / (1 - d)) duplicates, so duplicates are a share d of the reads, and they are left in
the data (not marked; a pipeline that marks and removes them sees only a lower depth, which the depth law covers). A
duplicate repeats its parent's mapping, retention and alleles, including the parent's errors (a PCR or sequencing
error in the parent molecule repeats), and carries its own independent errors. Depth is the configured depth of
unique fragments plus the duplicates. Long reads and independent sites refuse the setting.

Library error factors (library_error_sigma > 0): libraries differ in DNA damage, PCR cycles and run quality, so each
draws one factor f = exp(sigma Z - sigma^2 / 2) (mean one; stream simulation.library_error keyed by the generating seed
and the sample identifier, the same on every contig and at any sample order). Its reads carry error_rate x f
(paired-end and independent-site samplers; the long-read quality law is not scaled), and the call-set layer converts
its REF reads at error-prone sites with the site's rate x f (simulation/artefacts.py). Only the increase is clipped:
a rate r becomes min(r f, max(r, LIBRARY_ERROR_RATE_LIMIT)), so a configured rate above the limit is never lowered.
Nothing is drawn when sigma is zero.

Chimeric long reads (chimera_share > 0): a read is chimeric with that probability, joined to a partner molecule of the
same library (length from the read-length law; cross-sample joins are the foreign-read mode's). The aligner maps each
segment to its own locus (primary and supplementary alignments) and soft-clips a segment too short to place: when the
read is the shorter segment and below chimera_min_segment_bp it is lost; with chimera_mismap_share it is instead
aligned contiguously at the partner's locus (a uniform position) and reports alleles unrelated to the local genotype
(the homologue's allele at a uniform other site). The draws come from the read batch's stream after the donor draws,
only when the share is positive.

Collapsed paralogs (simulation/structure.py, collapsed_fraction > 0): each copy of a collapsed segment that a sample
carries adds one homologue's depth (depth x factor / 2, times the sample's site and regional depth factors, which a
near-identical copy shares) of reads lying inside the segment, with the sampler's geometry, clipped to the segment,
carrying the copy's base at every site, under the same read-level mapping loss as own reads and with no sequencing
error. Neither the count dispersion (gamma cell factors) nor the dropout tract acts on them, so a dropout tract inside
a segment shows only copy reads (both default off). They come from the stream simulation.sv.collapsed (read seed,
"reads", sample index) after the exogenous reads.

Structure (simulation/structure.py): a homologue inside a deletion it carries gives no reads (the long-read sampler
drops its hits; the other samplers' reads at such cells are assigned to homologues at 1/2 and redrawn per homologue),
and each carried duplication copy adds Poisson reads of its own allele at the per-homologue expected depth. These
draws use one stream per sample, simulation.sv.reads, after the sampler's own draws.

Random streams: the factors shared by all samples (sample depth factors, site factors)
and one seed come from the contig's stream; each sample's own draws (cell factors, depth or
fragments, dropout tract, alleles) come from its own stream, simulation.sample_reads(seed, sample
index). Blocks of samples are therefore simulated in parallel with the same result at any worker
count, because each sample stream is indexed independently of its execution block.

Batches (both read samplers): a sample's reads are drawn in batches of consecutive reads, at most READ_BATCH_READS
and, in expectation, READ_BATCH_INCIDENCES read-site incidences each, so the working memory is bounded at any depth.
The read count (or the per-window counts) comes first from the sample's stream; each read's start (within its window),
length, homologue, quality, foreign donor and thinning mark, and each incidence's draws, are independent across reads
and incidences, so batching changes no law. The first batch draws from the sample's stream in the order above, so a
sample whose reads fit in one batch draws exactly as unbatched; later batches draw from simulation.read_batch(seed,
sample index, batch); the dropout tract is drawn from the sample's stream after the first batch. The batch size is
fixed by the read model and the contig, so the counts are the same at any worker count.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri

from ..core.errors import SettingsError
from ..core.random import stream
from .design import ReadModel
from .parameters import (
    CONTAMINATION_SHARE_LIMIT,
    DUPLICATE_RATE_LIMIT,
    LIBRARY_ERROR_RATE_LIMIT,
    EXOGENOUS_SHARE_LIMIT,
    FRAGMENT_QUANTILES,
    FRAGMENT_SD_LIMIT,
    LONG_READ_MIN_BP,
    LONG_READ_SIGMA_LIMIT,
    MINIMUM_FRAGMENT_BP,
    READ_BATCH_INCIDENCES,
    READ_BATCH_READS,
    READ_QUALITY_MAX,
    READ_QUALITY_MIN,
)
from .structure import StructureReads, structural_counts, sv_stream

# Fragment-length quantiles used to compute the expected bases covered per fragment.


@dataclass(frozen=True)
class ReadInputs:
    """Everything one contig's per-sample read draws read: haplotypes int8[N, L], sorted positions
    int64[L], the sample depth factors float64[N] and site factors float64[L] (or None) drawn once for all samples,
    and the seed of the per-sample streams. Shared read-only with the workers simulating blocks of samples."""

    hap0: NDArray[np.int8]
    hap1: NDArray[np.int8]
    positions: NDArray[np.int64]
    factor: NDArray[np.float64]
    site_factor: NDArray[np.float64] | None
    depth: float
    model: ReadModel
    seed: int
    # Long-read mode: int8 [L], -1 = ordinary site, else the allele extending a homopolymer (context site).
    context: NDArray[np.int8] | None = None
    # Long-read mode: float64 [L], each context site's indel multiplier over context_factor (mean-one exponential).
    context_strength: NDArray[np.float64] | None = None
    # Structural variants of the analysed haplotypes (simulation/structure.py), or None.
    structure: StructureReads | None = None
    # Exogenous reads: the other species' base int8[L] (-1 outside conserved windows) and each sample's share
    # float64[N] of its mapped reads (None: every sample at exogenous_read_share); None when off.
    exogenous_allele: NDArray[np.int8] | None = None
    exogenous_share: NDArray[np.float64] | None = None
    # Collapsed paralogs (simulation/structure.py): segments int64[P] (half-open bp), each sample's copies int8[N, P]
    # and the copies' base int8[L] (-1 outside every segment); None when off.
    collapsed_start: NDArray[np.int64] | None = None
    collapsed_end: NDArray[np.int64] | None = None
    collapsed_copies: NDArray[np.int8] | None = None
    collapsed_allele: NDArray[np.int8] | None = None
    # Unsequenced contamination donors' collapsed copies int8[M, P], or None.
    collapsed_donor_copies: NDArray[np.int8] | None = None
    # Per-sample cross-sample contamination: each library's share float64[N] of reads from its donor and the donor's
    # row int64[N] (-1 clean); None when off.
    contamination_share: NDArray[np.float64] | None = None
    contamination_donor: NDArray[np.int64] | None = None
    # Unsequenced contamination donors' haplotypes int8[M, L] (donor rows N..N+M-1), or None.
    donor_hap0: NDArray[np.int8] | None = None
    donor_hap1: NDArray[np.int8] | None = None
    # PCR duplicates (paired-end): each library's duplicate rate float64[N]; None when off.
    duplicate_rate: NDArray[np.float64] | None = None
    # Library error factors float64[N] (mean one); None when off.
    error_factor: NDArray[np.float64] | None = None
    # Gate G1i (simulation/injection.py): the haplotypes before replacement and the replaced cells; None when clean.
    original0: NDArray[np.int8] | None = None
    original1: NDArray[np.int8] | None = None
    injected: NDArray[np.bool_] | None = None


def read_inputs(
    hap0: NDArray[np.int8],
    hap1: NDArray[np.int8],
    positions: NDArray[np.int64],
    depth: float,
    model: ReadModel,
    rng: np.random.Generator,
) -> ReadInputs:
    """Draw the factors shared by all samples and the per-sample stream seed from the contig's stream ``rng``."""
    hap0 = np.ascontiguousarray(hap0, dtype=np.int8)
    hap1 = np.ascontiguousarray(hap1, dtype=np.int8)
    positions = np.ascontiguousarray(positions, dtype=np.int64)
    samples, sites = hap0.shape
    if model.read_length_bp and (positions.shape != (sites,) or np.any(np.diff(positions) < 0)):
        raise SettingsError("fragment sampling needs sorted positions aligned with the haplotypes")
    factor = np.ascontiguousarray(_sample_factors(samples, model, rng), dtype=np.float64)
    site_factor = (
        _site_factors(positions, model, rng)
        if model.site_depth_cv or model.site_loss_fraction or model.site_loss_fine_fraction else None
    )
    if site_factor is not None:
        site_factor = np.ascontiguousarray(site_factor, dtype=np.float64)
    seed = int(rng.integers(0, 2**63))
    context = strength = None
    if model.long_read_mean_bp:
        # Drawn after every released draw of this stream, so the other samplers are unchanged.
        at = rng.random(sites) < model.context_sites
        allele = rng.integers(0, 2, size=sites)
        context = np.where(at, allele, -1).astype(np.int8)
        strength = rng.exponential(1.0, size=sites)
    exogenous = None
    if model.exogenous_read_share:
        # Drawn after every other draw of this stream, so a configuration without exogenous reads is unchanged.
        windows, inverse = np.unique(positions // model.exogenous_window_bp, return_inverse=True)
        conserved = rng.random(len(windows)) < model.exogenous_window_fraction
        base = (rng.random(sites) < model.exogenous_alt_probability).astype(np.int8)
        exogenous = np.ascontiguousarray(np.where(conserved[inverse], base, -1).astype(np.int8))
    return ReadInputs(
        hap0, hap1, positions, factor, site_factor, float(depth), model, seed, context, strength,
        exogenous_allele=exogenous,
    )


def exogenous_shares(sample_ids, model, seed) -> NDArray[np.float64]:
    """Each sample's genome-wide exogenous share (module docstring): contaminated with probability
    exogenous_sample_fraction, at exogenous_read_share x Exponential(1) clipped to EXOGENOUS_SHARE_LIMIT, from the
    stream simulation.exogenous_level keyed by the generating seed and the sample identifier (the same on every
    contig and at any sample order)."""
    shares = np.zeros(len(sample_ids))
    for row, sample in enumerate(sample_ids):
        g = stream("simulation.exogenous_level", seed=seed, key=str(sample))
        contaminated = g.random() < model.exogenous_sample_fraction
        level = model.exogenous_read_share * g.exponential(1.0)
        shares[row] = min(level, EXOGENOUS_SHARE_LIMIT) if contaminated else 0.0
    return shares


def contamination_levels(sample_ids, model, seed, extra_donors=()) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Each library's contamination share and donor row (module docstring), the same on every contig and at any sample
    order. An explicit list (contamination_samples, contamination_shares, optional contamination_donors) sets them
    exactly; a listed donor may be an unsequenced individual of an explicit pedigree (extra_donors, rows N..N+M-1 of
    the donor haplotypes). Otherwise the law: a library is contaminated with probability
    contamination_sample_fraction at contamination_share x Exponential(1), clipped at CONTAMINATION_SHARE_LIMIT, from
    one donor drawn uniformly among the other sequenced samples (stream simulation.contamination_level, keyed by the
    generating seed and the sample identifier; donors indexed in sorted identifier order)."""
    names = [str(s) for s in sample_ids]
    row = {s: i for i, s in enumerate(names)}
    shares = np.zeros(len(names))
    donors = np.full(len(names), -1, dtype=np.int64)
    ordered = sorted(names)
    if model.contamination_samples:
        donor_row = dict(row, **{str(d): len(names) + k for k, d in enumerate(extra_donors)})
        unknown = [s for s in model.contamination_samples if s not in row]
        unknown += [d for d in model.contamination_donors if d not in donor_row]
        if unknown:
            raise SettingsError(f"contamination names samples that are not sequenced: {', '.join(unknown[:5])}")
        for k, sample in enumerate(model.contamination_samples):
            shares[row[sample]] = model.contamination_shares[k]
            if model.contamination_donors:
                donors[row[sample]] = donor_row[model.contamination_donors[k]]
            else:
                g = stream("simulation.contamination_level", seed=seed, key=sample)
                others = [s for s in ordered if s != sample]
                donors[row[sample]] = row[others[int(g.random() * len(others))]]
        return shares, donors
    if not model.contamination_share or len(names) < 2:
        return shares, donors
    for sample in names:
        g = stream("simulation.contamination_level", seed=seed, key=sample)
        contaminated = g.random() < model.contamination_sample_fraction
        level = model.contamination_share * g.exponential(1.0)
        others = [s for s in ordered if s != sample]
        donor = others[int(g.random() * len(others))]
        if contaminated and level > 0:
            shares[row[sample]] = min(level, CONTAMINATION_SHARE_LIMIT)
            donors[row[sample]] = row[donor]
    return shares, donors


def duplicate_rates(sample_ids, model, seed) -> NDArray[np.float64]:
    """Each library's PCR duplicate rate: duplicate_rate x Exponential(1), clipped at DUPLICATE_RATE_LIMIT, from the
    stream simulation.duplicate_level keyed by the generating seed and the sample identifier (the same on every contig
    and at any sample order)."""
    rates = np.zeros(len(sample_ids))
    for row, sample in enumerate(sample_ids):
        g = stream("simulation.duplicate_level", seed=seed, key=str(sample))
        rates[row] = min(model.duplicate_rate * g.exponential(1.0), DUPLICATE_RATE_LIMIT)
    return rates


def library_error_factors(sample_ids, model, seed) -> NDArray[np.float64]:
    """Each library's error factor exp(sigma Z - sigma^2 / 2) (lognormal, mean one, sigma = library_error_sigma), from
    the stream simulation.library_error keyed by the generating seed and the sample identifier (the same on every
    contig and at any sample order)."""
    sigma = float(model.library_error_sigma)
    factors = np.ones(len(sample_ids))
    for row, sample in enumerate(sample_ids):
        g = stream("simulation.library_error", seed=seed, key=str(sample))
        factors[row] = math.exp(sigma * float(g.standard_normal()) - sigma * sigma / 2)
    return factors


OWN, LIBRARY_DONOR = -1, -2  # donor codes of a read: its own molecule, or the contaminated library's one donor


def _donor_alleles(donor, site, side, cohort, contamination):
    """(alleles, heterozygous) int64/bool of donor hits: a row >= 0 is a sequenced sample (uniform foreign reads),
    LIBRARY_DONOR the library's donor, whose two haplotype rows travel in ``contamination`` (only those rows enter a
    read block, sequenced or not)."""
    alleles = np.empty(len(site), dtype=np.int64)
    het = np.empty(len(site), dtype=bool)
    library = donor == LIBRARY_DONOR
    if library.any():
        d0, d1 = contamination[2], contamination[3]
        s, o = site[library], side[library]
        alleles[library] = np.where(o == 0, d0[s], d1[s])
        het[library] = d0[s] != d1[s]
    uniform = ~library
    if uniform.any():
        d, s, o = donor[uniform], site[uniform], side[uniform]
        alleles[uniform] = np.where(o == 0, cohort[0][d, s], cohort[1][d, s])
        het[uniform] = cohort[0][d, s] != cohort[1][d, s]
    return alleles, het


def _place_reads(g, k, lo, span, windows):
    """(starts, envelopes or None) of k new reads drawn as the sampler draws its own: uniform over the span, or with
    target windows a window chosen in proportion to its envelope x width, uniform within it, with its envelope."""
    if not windows:
        return lo + g.integers(0, span, size=k), None
    weight = np.cumsum(windows["local"] * windows["widths"])
    w = np.minimum(np.searchsorted(weight, g.random(k) * weight[-1], side="right"), len(weight) - 1)
    return windows["edges"][w] + np.floor(g.random(k) * windows["widths"][w]).astype(np.int64), windows["local"][w]


def _chimera_split(g, start, length, which, error, donor, envelope, model, lo, span, windows):
    """Chimeric long reads (module docstring): each chimeric read of length L is split at a uniform junction. One
    piece stays at the read's locus; the partner piece is placed at a uniform locus (drawn as the sampler draws reads)
    with its own homologue, so coverage is conserved. The shorter piece is lost below chimera_min_segment_bp, or with
    chimera_mismap_share aligned contiguously after the longer piece, where it reports unrelated alleles. Returns the
    extended read arrays and the mismapped mark."""
    m = len(start)
    mismapped = np.zeros(m, dtype=bool)
    chimeric = np.flatnonzero(g.random(m) < model.chimera_share)
    k = len(chimeric)
    if not k:
        return start, length, which, error, donor, envelope, mismapped
    total = length[chimeric]
    own = np.minimum(np.maximum(1, np.rint(g.random(k) * total)).astype(np.int64), total)
    part = total - own
    p_start, p_envelope = _place_reads(g, k, lo, span, windows)
    p_which = g.integers(0, 2, size=k)
    shorter_own = own < part
    short = np.minimum(own, part) < model.chimera_min_segment_bp
    mis = short & (g.random(k) < model.chimera_mismap_share) if model.chimera_mismap_share else np.zeros(k, bool)
    lost = short & ~mis
    a_start, a_len, a_mis = start[chimeric].copy(), np.where(lost & shorter_own, 0, own), mis & shorter_own
    b_start, b_len, b_mis = p_start.copy(), np.where(lost & ~shorter_own, 0, part), mis & ~shorter_own
    # A mismapped shorter piece is aligned right after the longer piece, at that piece's locus and envelope.
    a_start = np.where(a_mis, b_start + part, a_start)
    b_start = np.where(b_mis, start[chimeric] + own, b_start)
    start, length = start.copy(), length.copy()
    start[chimeric], length[chimeric], mismapped[chimeric] = a_start, a_len, a_mis
    start = np.concatenate((start, b_start))
    length = np.concatenate((length, b_len))
    which = np.concatenate((which, p_which))
    error = np.concatenate((error, error[chimeric]))
    mismapped = np.concatenate((mismapped, b_mis))
    if donor is not None:
        donor = np.concatenate((donor, donor[chimeric]))
    if envelope is not None:
        a_env = np.where(a_mis, p_envelope, envelope[chimeric])
        b_env = np.where(b_mis, envelope[chimeric], p_envelope)
        envelope = envelope.copy()
        envelope[chimeric] = a_env
        envelope = np.concatenate((envelope, b_env))
    return start, length, which, error, donor, envelope, mismapped


def _read_donors(g, m, which, model, cohort, contamination):
    """int64 [m]: each read's (or fragment's) donor row, -1 for the sample's own molecules; None when no read can be
    foreign. Uniform foreign reads (foreign_read_share, a uniform other sample per read) are drawn first, exactly as
    the long-read sampler always drew them; a contaminated library's reads then come from its one donor with
    probability its share (drawn only when the share is positive). The donor's homologue is the read's ``which``."""
    donor = None
    if model.foreign_read_share and cohort is not None and len(cohort[0]) > 1:
        foreign = g.random(m) < model.foreign_read_share
        other = g.integers(0, len(cohort[0]) - 1, size=m)
        donor = np.where(foreign, other + (other >= cohort[2]), OWN)
    if contamination is not None and contamination[0] > 0:
        mixed = g.random(m) < contamination[0]
        donor = np.where(mixed, LIBRARY_DONOR, OWN if donor is None else donor)
    return donor


def read_rows(inputs: ReadInputs, first: int, stop: int, out: NDArray[np.integer] | None = None) -> NDArray[np.integer]:
    """REF/ALT counts int64[stop - first, L, 2] of samples first..stop-1, each from its own stream
    (simulation.sample_reads, the inputs' seed and the sample's index): any partition of the samples into blocks,
    simulated anywhere, gives the same counts. ``out`` (optional, integer [stop - first, L, 2]): written
    in place and returned instead (the simulation stage's shared int32 counts)."""
    model = inputs.model
    sample = _fragment_sample if model.read_length_bp else _site_sample
    if model.long_read_mean_bp:
        sample = _long_read_sample
    sites = inputs.hap0.shape[1]
    if out is None:
        out = np.zeros((stop - first, sites, 2), dtype=np.int64)
    elif out.shape != (stop - first, sites, 2):
        raise ValueError("read counts destination has the wrong shape")
    n_sequenced = len(inputs.hap0)
    for row, i in enumerate(range(first, stop)):
        rng = stream("simulation.sample_reads", seed=inputs.seed, index=i)
        model = inputs.model
        if inputs.error_factor is not None:
            # The library's error susceptibility scales its base error rate (the long-read quality law is not scaled).
            # Only the increase is clipped: a configured rate above the limit is never lowered.
            model = replace(model, error_rate=min(model.error_rate * float(inputs.error_factor[i]),
                                                  max(model.error_rate, LIBRARY_ERROR_RATE_LIMIT)))
        args = (inputs.hap0[i], inputs.hap1[i], inputs.positions, inputs.depth, model, float(inputs.factor[i]))
        target = inputs.site_factor
        if inputs.model.region_depth_cv:
            # The regional factors have their own stream per sample, so the reads are sampled once and a
            # zero regional CV leaves the sample's read stream, and so every count, as it was.
            regional = _region_factors(
                inputs.positions, inputs.model, stream("simulation.sample_region", seed=inputs.seed, index=i)
            )
            target = regional if target is None else target * regional
        structure = inputs.structure
        cohort = (inputs.hap0, inputs.hap1, i)
        contamination = None
        share = 0.0
        if inputs.contamination_share is not None and inputs.contamination_share[i] > 0:
            # Only the donor's two haplotype rows enter the block (a sequenced sample or an unsequenced individual).
            share, d = float(inputs.contamination_share[i]), int(inputs.contamination_donor[i])
            rows = (inputs.hap0[d], inputs.hap1[d]) if d < n_sequenced else (
                inputs.donor_hap0[d - n_sequenced], inputs.donor_hap1[d - n_sequenced])
            contamination = (share, d, rows[0], rows[1])
        duplicate = 0.0 if inputs.duplicate_rate is None else float(inputs.duplicate_rate[i])
        present = None if structure is None else structure.present[i]
        # Own reads leave a carried deletion inside the paired-end sampler whenever other molecules (donor reads,
        # duplicates) must keep their own fate there; otherwise the released post-hoc thinning applies unchanged.
        in_sampler = bool(
            model.read_length_bp and present is not None and not present.all()
            and (model.foreign_read_share or contamination is not None or duplicate > 0)
        )
        if model.long_read_mean_bp:
            context = None if inputs.context is None else (inputs.context, inputs.context_strength)
            counts = sample(*args, target, rng, context, present, cohort, contamination=contamination,
                            batch_key=(inputs.seed, i))
        elif model.read_length_bp:
            counts = sample(*args, target, rng, cohort=cohort, contamination=contamination, duplicate=duplicate,
                            present=present if in_sampler else None, batch_key=(inputs.seed, i))
        else:
            counts = sample(*args, target, rng)
        if structure is not None and (not structure.present[i].all() or (structure.copies[i] >= 0).any()):
            # Expected reads of one homologue copy (unimpaired depth x sample, site and regional factors); a
            # contaminated library's own copies are the (1 - share) of its molecules that are its own.
            lam = np.full(sites, inputs.depth * float(inputs.factor[i]) / 2 * (1.0 - share))
            if target is not None:
                lam = lam * target
            error = _mean_other_allele_rate(model) if model.long_read_mean_bp else model.error_rate
            counts = structural_counts(
                counts,
                np.stack((inputs.hap0[i], inputs.hap1[i])),
                structure.present[i],
                structure.copies[i],
                lam,
                error,
                sv_stream("simulation.sv.reads", inputs.seed, i),
                thin=not model.long_read_mean_bp and not in_sampler,
                duplicate=duplicate,
            )
        if contamination is not None and not model.read_length_bp and not model.long_read_mean_bp:
            # Independent sites: donor reads replace own reads after the structure, so a recipient's deletion keeps
            # the donor's reads (drawn from the sample's stream after its other draws).
            counts = _site_contamination(counts, contamination, model, rng)
        if inputs.exogenous_allele is not None:
            exo = model.exogenous_read_share if inputs.exogenous_share is None else float(inputs.exogenous_share[i])
            if exo > 0:
                counts = counts + _exogenous_counts(
                    inputs.positions, inputs.exogenous_allele, inputs.depth * float(inputs.factor[i]), exo, model,
                    stream("simulation.exogenous_reads", seed=inputs.seed, index=i), duplicate=duplicate,
                )
        if inputs.collapsed_copies is not None and inputs.collapsed_copies[i].any():
            counts = counts + _collapsed_counts(
                inputs.positions, inputs.collapsed_allele, inputs.collapsed_start, inputs.collapsed_end,
                inputs.collapsed_copies[i], inputs.depth * float(inputs.factor[i]) / 2 * (1.0 - share), model,
                sv_stream("simulation.sv.collapsed", inputs.seed, "reads", i), target, duplicate=duplicate,
            )
        if contamination is not None and inputs.collapsed_copies is not None:
            # The donor's own collapsed copies, at the share of the library that is the donor's.
            d = contamination[1]
            donor_copies = inputs.collapsed_copies[d] if d < n_sequenced else (
                None if inputs.collapsed_donor_copies is None else inputs.collapsed_donor_copies[d - n_sequenced])
            if donor_copies is not None and donor_copies.any():
                counts = counts + _collapsed_counts(
                    inputs.positions, inputs.collapsed_allele, inputs.collapsed_start, inputs.collapsed_end,
                    donor_copies, inputs.depth * float(inputs.factor[i]) / 2 * share, model,
                    sv_stream("simulation.sv.collapsed", inputs.seed, "donor-reads", i), target, duplicate=duplicate,
                )
        if model.uncalled_base_probability:
            counts -= rng.binomial(counts, model.uncalled_base_probability)
        out[row] = counts
    return out


def _exogenous_counts(positions, allele, depth, share, model, rng, duplicate=0.0):
    """REF/ALT counts int64[L, 2] of one sample's exogenous reads (module docstring), from its own stream. duplicate:
    the library's PCR duplicate rate; each exogenous read then has Poisson(d / (1 - d)) duplicates (paired-end)."""
    sites = len(positions)
    out = np.zeros((sites, 2), dtype=np.int64)
    inside = allele >= 0
    if not inside.any():
        return out
    width = int(model.exogenous_window_bp)
    windows = np.unique(positions[inside] // width)
    conserved_bp = len(windows) * width
    span = int(positions[-1]) - int(positions[0]) + 1  # the sites' span, the genome the endogenous reads cover
    exo_depth = depth * share / (1.0 - share) * span / conserved_bp
    if not model.read_length_bp and not model.long_read_mean_bp:
        n = rng.poisson(exo_depth, size=sites) * inside
        out[:, 1] = np.where(allele == 1, n, 0)
        out[:, 0] = n - out[:, 1]
        return out
    if model.long_read_mean_bp:
        mu, sigma, f_min, f_max, covered = _long_read_lengths(model)
    else:
        r, mean, sd = int(model.read_length_bp), float(model.fragment_mean_bp), float(model.fragment_sd_bp)
        f_max = int(math.ceil(mean + FRAGMENT_SD_LIMIT * sd)) + 1
        quantiles = (np.arange(FRAGMENT_QUANTILES) + 0.5) / FRAGMENT_QUANTILES
        lengths = np.clip(np.rint(mean + sd * ndtri(quantiles)), MINIMUM_FRAGMENT_BP, f_max)
        covered = float(np.mean(np.minimum(lengths, 2 * r)))
    # Starts are uniform over [window start - f_max, window end), so every position of a window has the same expected
    # depth (n covered / (width + f_max) per window, as in the samplers); a read keeps only its part inside the window.
    n = int(rng.poisson(exo_depth * len(windows) * (width + f_max) / covered))
    if n == 0:
        return out
    lo = windows[rng.integers(0, len(windows), size=n)] * width
    start = lo - f_max + rng.integers(0, width + f_max, size=n)
    hi = lo + width
    if model.long_read_mean_bp:
        length = np.clip(np.rint(np.exp(rng.normal(mu, sigma, size=n))), f_min, f_max).astype(np.int64)
        ends = [(start, start + length)]
    else:
        length = np.clip(np.rint(rng.normal(mean, sd, size=n)), MINIMUM_FRAGMENT_BP, f_max).astype(np.int64)
        first_end = start + np.minimum(length, r)
        ends = [(start, first_end), (np.maximum(start + length - r, first_end), start + length)]
    spans = [(np.clip(a, lo, hi), np.clip(b, lo, hi)) for a, b in ends]
    depth_counts = np.zeros(sites, dtype=np.int64)
    copies = rng.poisson(duplicate / (1.0 - duplicate), size=n) if duplicate else None
    for a, b in spans:
        first, stop = np.searchsorted(positions, a), np.searchsorted(positions, b)
        size = np.maximum(stop - first, 0)
        if not size.any():
            continue
        offsets = np.arange(int(size.sum())) - np.repeat(np.cumsum(size) - size, size)
        hits = np.repeat(first, size) + offsets
        if copies is not None:
            hits = np.concatenate((hits, np.repeat(hits, np.repeat(copies, size))))
        depth_counts += np.bincount(hits, minlength=sites)
    out[:, 1] = np.where(allele == 1, depth_counts, 0)
    out[:, 0] = depth_counts - out[:, 1]
    return out


def _collapsed_counts(positions, allele, start, end, copies, homologue_depth, model, rng, target=None, duplicate=0.0):
    """REF/ALT counts int64[L, 2] of one sample's reads from collapsed-paralog copies (structure.py), from its own
    stream. Each carried copy of a segment adds one homologue's depth of reads lying inside the segment, with the
    sampler's read geometry (independent sites, paired mates or long reads) and the copy's base at every site.

    target (float64[L] or None): the sample's site x regional depth factors, as for its own reads. A near-identical
    copy shares the segment's mappability, so its reads take the same factors: the independent-site rate is
    multiplied by them; read geometries are drawn at each segment's largest factor and each read's site hits kept
    with one uniform per read (hit kept when the uniform is below factor / segment maximum), the fragment sampler's
    construction. The read-level mapping loss applies as to own reads (module docstring), with k the copy's ALT
    alleles under the mate (paired) or within reference_bias_window_bp of the site (long reads). duplicate: the
    library's PCR duplicate rate (paired-end); each kept copy read has Poisson(d / (1 - d)) duplicates repeating its
    hits (copy reads carry no error), drawn after its other draws."""
    sites = len(positions)
    out = np.zeros((sites, 2), dtype=np.int64)
    segment = np.searchsorted(start, positions, side="right") - 1
    inside = (segment >= 0) & (positions < end[np.maximum(segment, 0)])
    if not model.read_length_bp and not model.long_read_mean_bp:
        rate = np.where(inside, copies[np.maximum(segment, 0)] * homologue_depth, 0.0)
        if target is not None:
            rate = rate * target
        n = rng.poisson(rate)
        out[:, 1] = np.where(allele == 1, n, 0)
        out[:, 0] = n - out[:, 1]
        return out
    if model.long_read_mean_bp:
        mu, sigma, f_min, f_max, covered = _long_read_lengths(model)
    else:
        r, mean, sd = int(model.read_length_bp), float(model.fragment_mean_bp), float(model.fragment_sd_bp)
        f_max = int(math.ceil(mean + FRAGMENT_SD_LIMIT * sd)) + 1
        quantiles = (np.arange(FRAGMENT_QUANTILES) + 0.5) / FRAGMENT_QUANTILES
        lengths = np.clip(np.rint(mean + sd * ndtri(quantiles)), MINIMUM_FRAGMENT_BP, f_max)
        covered = float(np.mean(np.minimum(lengths, 2 * r)))
    # Each segment's largest depth factor (1 without factors): reads are drawn at it and thinned per read below.
    peak = np.ones(len(start))
    if target is not None:
        peak = np.zeros(len(start))
        np.maximum.at(peak, segment[inside], target[inside])
    # As for exogenous reads: starts uniform over [segment start - f_max, segment end) give every position of the
    # segment the same expected depth; a read keeps only its part inside the segment.
    widths = (end - start).astype(np.float64)
    n = rng.poisson(copies * homologue_depth * peak * (widths + f_max) / covered)
    total = int(n.sum())
    if total == 0:
        return out
    which = np.repeat(np.arange(len(start)), n)
    lo, hi = start[which], end[which]
    first_start = lo - f_max + np.floor(rng.random(total) * (hi - lo + f_max)).astype(np.int64)
    copy_alt = allele == 1
    if model.long_read_mean_bp:
        length = np.clip(np.rint(np.exp(rng.normal(mu, sigma, size=total))), f_min, f_max).astype(np.int64)
        a = np.searchsorted(positions, np.clip(first_start, lo, hi))
        b = np.searchsorted(positions, np.clip(first_start + length, lo, hi))
        spans = [(a, b)]
    else:
        length = np.clip(np.rint(rng.normal(mean, sd, size=total)), MINIMUM_FRAGMENT_BP, f_max).astype(np.int64)
        first_end = first_start + np.minimum(length, r)
        second_from = np.maximum(first_start + length - r, first_end)
        a1 = np.searchsorted(positions, np.clip(first_start, lo, hi))
        b1 = np.searchsorted(positions, np.clip(first_end, lo, hi))
        a2 = np.searchsorted(positions, np.clip(second_from, lo, hi))
        b2 = np.searchsorted(positions, np.clip(first_start + length, lo, hi))
        if model.reference_bias:
            # The paired-end mapping loss of own reads (_mate_mapping_loss), on the copy's alleles.
            prefix = np.r_[0, np.cumsum(copy_alt, dtype=np.int64)]
            second_full = np.searchsorted(positions, np.clip(np.maximum(first_start + length - r, first_start), lo, hi))
            k = np.stack((prefix[b1] - prefix[a1], prefix[b2] - prefix[second_full]), axis=1).astype(np.float64)
            lost = rng.random(k.shape) < -np.expm1(-model.reference_bias * k**model.reference_bias_power)
            a2 = np.where(lost[:, 0], second_full, a2)
            b1 = np.where(lost[:, 0], a1, b1)
            b2 = np.where(lost[:, 1], a2, b2)
        spans = [(a1, b1), (a2, b2)]
    site_parts, read_parts = [], []
    for first, stop in spans:
        size = np.maximum(stop - first, 0)
        if not size.any():
            continue
        offsets = np.arange(int(size.sum())) - np.repeat(np.cumsum(size) - size, size)
        site_parts.append(np.repeat(first, size) + offsets)
        read_parts.append(np.repeat(np.arange(total), size))
    if not site_parts:
        return out
    site, read = np.concatenate(site_parts), np.concatenate(read_parts)
    if target is not None:
        mark = rng.random(total)
        keep = mark[read] < target[site] / peak[which[read]]
        site, read = site[keep], read[keep]
    if model.long_read_mean_bp and model.reference_bias:
        # The long-read local loss of own reads: a base is lost with 1 - exp(-b k^p), k the copy's ALT alleles within
        # reference_bias_window_bp of the site.
        k = _window_alt_counts(copy_alt.astype(np.int8), positions, int(model.reference_bias_window_bp))[site]
        hazard = -np.expm1(-model.reference_bias * k.astype(np.float64) ** model.reference_bias_power)
        kept = rng.random(len(site)) >= hazard
        site, read = site[kept], read[kept]
    if duplicate:
        site = np.concatenate((site, np.repeat(site, rng.poisson(duplicate / (1.0 - duplicate), size=total)[read])))
    depth_counts = np.bincount(site, minlength=sites).astype(np.int64)
    out[:, 1] = np.where(allele == 1, depth_counts, 0)
    out[:, 0] = depth_counts - out[:, 1]
    return out


def _quality_quantiles(model):
    """Per-read error rates at fixed quantiles of the clipped Phred law (deterministic integration)."""
    q = (np.arange(FRAGMENT_QUANTILES) + 0.5) / FRAGMENT_QUANTILES
    quality = np.clip(model.read_quality_mean + model.read_quality_sd * ndtri(q), READ_QUALITY_MIN, READ_QUALITY_MAX)
    return 10.0 ** (-quality / 10)


def _mean_error(model) -> float:
    """Mean per-base error rate of a long read under the clipped Phred law."""
    return float(np.mean(_quality_quantiles(model)))


def _mean_other_allele_rate(model) -> float:
    """Mean probability that a long read reports the other allele at an ordinary site (substitutions only)."""
    return _mean_error(model) * model.substitution_share / 3


def _sample_factors(samples, model, rng):
    factor = np.ones(samples)
    if model.depth_cv:
        sigma = np.sqrt(np.log1p(model.depth_cv**2))
        factor = rng.lognormal(-sigma * sigma / 2, sigma, size=samples)
        factor /= factor.mean()
    return factor


def _site_factors(positions, model, rng):
    """Shared lognormal times independent coarse/fine losses, without renormalisation.

    Population mean is (1 - coarse_fraction / 2) * (1 - fine_fraction / 2).

    Physical windows are anchored at bp zero; only occupied windows consume draws.
    Window zero uses independent sites. Spatial sharing is separate from fragment sharing.
    """
    sites = len(positions)
    if not model.site_depth_cv and not model.site_loss_fraction and not model.site_loss_fine_fraction:
        return np.ones(sites)
    if model.site_depth_window_bp:
        windows, inverse = np.unique(np.asarray(positions) // model.site_depth_window_bp, return_inverse=True)
        size = len(windows)
    else:
        inverse, size = None, sites
    factor = np.ones(size)
    if model.site_depth_cv:
        sigma = np.sqrt(np.log1p(model.site_depth_cv**2))
        factor = rng.lognormal(-sigma * sigma / 2, sigma, size=size)
    if model.site_loss_fraction:
        impaired = rng.random(size) < model.site_loss_fraction
        retained = 1 - rng.random(size)
        factor *= np.where(impaired, retained, 1.0)
    factor = factor if inverse is None else factor[inverse]
    if model.site_loss_fine_fraction:
        windows, fine_inverse = np.unique(
            np.asarray(positions) // model.site_loss_fine_window_bp, return_inverse=True
        )
        impaired = rng.random(len(windows)) < model.site_loss_fine_fraction
        retained = 1 - rng.random(len(windows))
        factor *= np.where(impaired, retained, 1.0)[fine_inverse]
    return factor


def _region_factors(positions, model, rng):
    """Independent mean-one sample-window lognormal factors; zero CV consumes no draw.

    Windows share the site-factor origin at bp zero. Only occupied windows are
    drawn. No finite-contig or finite-sample normalisation changes the law.
    """
    if not model.region_depth_cv:
        return np.ones(len(positions))
    windows, inverse = np.unique(np.asarray(positions) // model.region_depth_window_bp, return_inverse=True)
    sigma = np.sqrt(np.log1p(model.region_depth_cv**2))
    return rng.lognormal(-sigma * sigma / 2, sigma, size=len(windows))[inverse]


def _fragment_envelopes(positions, target, lo, span, f_max):
    """Start windows of ``f_max`` bp over [lo, lo + span), with each window's thinning envelope.

    A fragment starting in window k covers sites only in [start, start + f_max), inside windows
    k and k + 1, so max(block maximum k, block maximum k + 1) bounds every target such a
    fragment can reach. Returns window starts, widths (the last may be shorter) and envelopes;
    a window that reaches no site has envelope zero and draws no fragment.
    """
    edges = lo + np.arange(0, span, f_max, dtype=np.int64)
    widths = np.minimum(f_max, lo + span - edges)
    first = np.searchsorted(positions, edges)
    stop = np.searchsorted(positions, edges + widths)
    block = np.zeros(len(edges))
    nonempty = stop > first
    if np.any(nonempty):
        # Blocks are disjoint and contiguous: each nonempty block runs to the next one's first site.
        block[nonempty] = np.maximum.reduceat(target, first[nonempty])
    local = block.copy()
    local[:-1] = np.maximum(block[:-1], block[1:])
    return edges, widths, local


def _site_sample(hap0, hap1, positions, depth, model, factor, site_factor, rng):
    """REF/ALT counts int64[L, 2] of one sample under the independent-site sampler, from its own stream (a
    contaminated library's donor reads are applied after the structure: _site_contamination)."""
    sites = len(hap0)
    width = int(sites * model.dropout_fraction)
    lam = np.full(sites, depth * factor / (1 - width / sites))
    if site_factor is not None:
        lam = lam * site_factor
    if model.count_dispersion:
        lam = lam * rng.gamma(1 / model.count_dispersion, model.count_dispersion, size=sites)
    counts = rng.poisson(lam=lam)
    if width:
        start = int(rng.integers(0, sites - width + 1))
        counts[start : start + width] = 0
    dosage = hap0.astype(np.int64) + hap1
    alt = np.zeros_like(counts)
    for genotype, probability in enumerate(
        (model.error_rate, model.heterozygote_alt_probability, 1 - model.error_rate)
    ):
        mask = dosage == genotype
        if np.any(mask):
            alt[mask] = rng.binomial(counts[mask], probability)
    return np.stack((counts - alt, alt), axis=-1).astype(np.int64, copy=False)


def _site_contamination(counts, contamination, model, rng):
    """Independent sites: each read of a contaminated library is replaced by a donor read with probability its share
    (binomial replacement, the kept own reads hypergeometric, donor reads at the donor's genotype law)."""
    share, _, d0, d1 = contamination
    counts = np.asarray(counts, dtype=np.int64)
    depth, alt = counts.sum(axis=1), counts[:, 1]
    replaced = rng.binomial(depth, share)
    own_alt = rng.hypergeometric(alt, depth - alt, depth - replaced)
    dosage = d0.astype(np.int64) + d1
    p_alt = np.choose(dosage, (model.error_rate, model.heterozygote_alt_probability, 1 - model.error_rate))
    alt = own_alt + rng.binomial(replaced, p_alt)
    return np.stack((depth - alt, alt), axis=-1)


def _batch_size(bases_per_read, sites, span):
    """Reads per batch of one sample (module docstring, Batches): at most READ_BATCH_READS, and READ_BATCH_INCIDENCES
    read-site incidences at the expected sites per read (covered bases per read x the contig's site density). Fixed by
    the read model and the contig, never by depth or the sample's draws."""
    per_read = max(float(bases_per_read) * sites / max(span, 1), 1.0)
    return int(max(1, min(READ_BATCH_READS, READ_BATCH_INCIDENCES // per_read)))


def _batches(rng, n, size, batch_key, edges=None, widths=None, local=None, window_ends=None):
    """Yield (generator, reads, start, envelope) per batch of a sample's n reads: the sample's stream for the first
    batch, simulation.read_batch(seed, sample index, batch) after it. Starts are uniform over the contig span (lo is
    added by the caller via edges=None) or, with target windows, uniform within each read's window (window_ends: the
    cumulative per-window read counts), with that window's envelope."""
    for b, a in enumerate(range(0, n, size)):
        g = rng if b == 0 else stream("simulation.read_batch", seed=batch_key[0], index=batch_key[1], key=b)
        m = min(n, a + size) - a
        if window_ends is None:
            yield g, m, None, None
        else:
            w = np.searchsorted(window_ends, np.arange(a, a + m), side="right")
            yield g, m, edges[w] + g.integers(0, widths[w]), local[w]


def _fragment_sample(hap0, hap1, positions, depth, model, factor, site_factor, rng, *, batch_key, cohort=None,
                     contamination=None, duplicate=0.0, present=None):
    """REF/ALT counts int64[L, 2] of one sample under the paired-end fragment sampler, from its own stream (later
    read batches from their own: module docstring, Batches). cohort = (hap0 [N, L], hap1 [N, L], this sample's row)
    and contamination = (share, donor row) or None: a foreign fragment is drawn whole, both mates, from the donor's
    homologue, with the sampler's geometry, mapping loss and thinning (_read_donors). duplicate: the library's PCR
    duplicate rate d; each sequenced fragment then has Poisson(d / (1 - d)) duplicates (a share d of all reads), left
    in: a duplicate repeats its parent's mapping, retention and alleles with the parent's errors, and adds its own
    independent errors (drawn after every other draw of the batch, only when d > 0). present: bool [2, L] or None;
    when given, the sample's own hits at a homologue's absent cells (a carried deletion) are removed here, donor hits
    kept, so duplicates and donor reads keep their fate there (read_rows then adds structure copies only)."""
    sites = len(hap0)
    r = int(model.read_length_bp)
    mean, sd = float(model.fragment_mean_bp), float(model.fragment_sd_bp)
    f_min, f_max = MINIMUM_FRAGMENT_BP, int(math.ceil(mean + FRAGMENT_SD_LIMIT * sd)) + 1
    # Expected bases covered per fragment under the sampled length law (normal, rounded, clipped)
    # evaluated at fixed quantiles.
    quantiles = (np.arange(FRAGMENT_QUANTILES) + 0.5) / FRAGMENT_QUANTILES
    lengths = np.clip(np.rint(mean + sd * ndtri(quantiles)), f_min, f_max)
    covered_per_fragment = float(np.mean(np.minimum(lengths, 2 * r)))
    width = int(sites * model.dropout_fraction)
    lo = int(positions[0]) - f_max
    span = int(positions[-1]) + 1 - lo
    counts = np.zeros(sites, dtype=np.int64)
    alt = np.zeros(sites, dtype=np.int64)
    rate = depth * factor / (1 - width / sites) / covered_per_fragment
    target = site_factor
    if model.count_dispersion:
        target = rng.gamma(1 / model.count_dispersion, model.count_dispersion, size=sites)
        if site_factor is not None:
            target *= site_factor
    if target is None:
        n = int(rng.poisson(rate * span))
        windows = {}
    else:
        # Conditional Poisson thinning gives each cell its target rate while retaining
        # common fragment starts/lengths/homologues across sites. Gamma mixing then
        # gives exact NB marginals, unlike duplicating already sampled read counts.
        # Starts are drawn per window of f_max bp at the window's own envelope (the
        # largest target any fragment starting there can reach), so candidate
        # fragments scale with local, not contig-wide, maxima of the target.
        edges, widths, local = _fragment_envelopes(positions, target, lo, span, f_max)
        per_window = rng.poisson(rate * local * widths)
        n = int(per_window.sum())
        windows = dict(edges=edges, widths=widths, local=local, window_ends=np.cumsum(per_window))
    # Reference mapping bias (module docstring): cumulative ALT counts of each homologue give a mate's k in O(1).
    alt_prefix = None
    if model.reference_bias:
        alt_prefix = (np.r_[0, np.cumsum(hap0, dtype=np.int64)], np.r_[0, np.cumsum(hap1, dtype=np.int64)])
    size_b = _batch_size(covered_per_fragment, sites, span)
    for g, m, start, envelope in _batches(rng, n, size_b, batch_key, **windows):
        if start is None:
            start = lo + g.integers(0, span, size=m)
        length = np.clip(np.rint(g.normal(mean, sd, size=m)), f_min, f_max).astype(np.int64)
        which = g.integers(0, 2, size=m)
        donor = _read_donors(g, m, which, model, cohort, contamination)
        first_end = start + np.minimum(length, r)
        second_start = np.maximum(start + length - r, first_end)
        second_end = start + length
        intervals = (
            (np.searchsorted(positions, start), np.searchsorted(positions, first_end)),
            (np.searchsorted(positions, second_start), np.searchsorted(positions, second_end)),
        )
        if alt_prefix is not None:
            intervals = _mate_mapping_loss(intervals, positions, start, length, r, which, alt_prefix, model, g,
                                           donor=donor, cohort=cohort, contamination=contamination)
        site_parts, hap_parts, fragment_parts = [], [], []
        for first, stop in intervals:
            size = stop - first
            keep = size > 0
            if not np.any(keep):
                continue
            first, size, owner = first[keep], size[keep], which[keep]
            offsets = np.arange(int(size.sum())) - np.repeat(np.cumsum(size) - size, size)
            site_parts.append(np.repeat(first, size) + offsets)
            hap_parts.append(np.repeat(owner, size))
            if target is not None or donor is not None or duplicate or present is not None:
                fragment_parts.append(np.repeat(np.flatnonzero(keep), size))
        if not site_parts:
            continue
        site = np.concatenate(site_parts)
        owner = np.concatenate(hap_parts)
        del site_parts, hap_parts
        fragment = np.concatenate(fragment_parts) if fragment_parts else None
        if target is not None:
            # One uniform mark per fragment couples retention across all its hits.
            # Shared-hit intensity is proportional to min(target_i, target_j), so an
            # unrelated high-depth site cannot change local dependence via the envelope.
            mark = g.random(m)
            keep = mark[fragment] < target[site] / envelope[fragment]
            site, owner, fragment = site[keep], owner[keep], fragment[keep]
            del keep
        if present is not None:
            here = present[owner, site] if donor is None else present[owner, site] | (donor[fragment] != OWN)
            site, owner, fragment = site[here], owner[here], fragment[here]
            del here
        allele = np.where(owner == 0, hap0[site], hap1[site]).astype(np.int64)
        het = hap0[site] != hap1[site]
        if donor is not None:
            outside = donor[fragment] != OWN
            allele[outside], het[outside] = _donor_alleles(
                donor[fragment][outside], site[outside], owner[outside], cohort, contamination)
        allele ^= (~het) & (g.random(len(site)) < model.error_rate)
        counts += np.bincount(site, minlength=sites).astype(np.int64)
        alt += np.bincount(site, weights=allele, minlength=sites).astype(np.int64)
        if duplicate:
            copies = g.poisson(duplicate / (1.0 - duplicate), size=m)[fragment]
            site_d, allele_d, het_d = np.repeat(site, copies), np.repeat(allele, copies), np.repeat(het, copies)
            allele_d ^= (~het_d) & (g.random(len(site_d)) < model.error_rate)
            counts += np.bincount(site_d, minlength=sites).astype(np.int64)
            alt += np.bincount(site_d, weights=allele_d, minlength=sites).astype(np.int64)
        del fragment
    if width:
        first = int(rng.integers(0, sites - width + 1))
        counts[first : first + width] = 0
        alt[first : first + width] = 0
    return np.stack((counts - alt, alt), axis=-1).astype(np.int64, copy=False)


def _donor_alt_counts(lo, hi, rows, which, cohort):
    """int64 [n]: ALT alleles of donor homologue (rows, which) over the site intervals [lo, hi)."""
    size = np.maximum(hi - lo, 0)
    total = int(size.sum())
    if total == 0:
        return np.zeros(len(lo), dtype=np.int64)
    offsets = np.arange(total) - np.repeat(np.cumsum(size) - size, size)
    site = np.repeat(lo, size) + offsets
    row, side = np.repeat(rows, size), np.repeat(which, size)
    values = np.where(side == 0, cohort[0][row, site], cohort[1][row, site])
    return np.bincount(np.repeat(np.arange(len(lo)), size), weights=values, minlength=len(lo)).astype(np.int64)


def _mate_mapping_loss(intervals, positions, start, length, r, which, alt_prefix, model, g, donor=None, cohort=None,
                       contamination=None):
    """Site intervals of the mates that align, under the read-level reference bias (module docstring).

    Each mate's k is the ALT count of the fragment's homologue over the sites of that mate's own span (the second
    mate's full span, overlap included); one uniform per mate decides, drawn from the batch's stream after the
    homologues. A lost mate contributes no site; when the first mate is lost the second keeps the overlap it reads.
    """
    (a1, b1), (a2, b2) = intervals
    second_full = np.searchsorted(positions, np.maximum(start + length - r, start))

    def alt_count(lo, hi):
        own = np.where(which == 0, alt_prefix[0][hi] - alt_prefix[0][lo], alt_prefix[1][hi] - alt_prefix[1][lo])
        if donor is not None and np.any(donor != OWN):
            # A foreign fragment's k counts its donor homologue's ALT alleles.
            sel = np.flatnonzero(donor >= 0)
            if len(sel):
                own[sel] = _donor_alt_counts(lo[sel], hi[sel], donor[sel], which[sel], cohort)
            lib = np.flatnonzero(donor == LIBRARY_DONOR)
            if len(lib):
                rows = (contamination[2][None, :], contamination[3][None, :])
                own[lib] = _donor_alt_counts(lo[lib], hi[lib], np.zeros(len(lib), np.int64), which[lib], rows)
        return own

    k = np.stack((alt_count(a1, b1), alt_count(second_full, b2)), axis=1).astype(np.float64)
    lost = g.random(k.shape) < -np.expm1(-model.reference_bias * k**model.reference_bias_power)
    a2 = np.where(lost[:, 0], second_full, a2)
    b1 = np.where(lost[:, 0], a1, b1)
    b2 = np.where(lost[:, 1], a2, b2)
    return (a1, b1), (a2, b2)


def _donor_window_alt(donor, site, side, cohort, library_alt, bounds):
    """int64 [n]: each donor hit's ALT alleles of its donor homologue within the reference-bias window: the library's
    donor from its precomputed window counts, a uniform foreign donor from that homologue's prefix sums."""
    k = np.empty(len(site), dtype=np.int64)
    library = donor == LIBRARY_DONOR
    if library.any():
        k[library] = np.where(side[library] == 0, library_alt[0][site[library]], library_alt[1][site[library]])
    for row, homologue in {(int(d), int(o)) for d, o in zip(donor[~library], side[~library], strict=True)}:
        hit = ~library & (donor == row) & (side == homologue)
        prefix = np.r_[0, np.cumsum(cohort[homologue][row], dtype=np.int64)]
        k[hit] = prefix[bounds[1][site[hit]]] - prefix[bounds[0][site[hit]]]
    return k


def _long_read_lengths(model):
    """(log-mean, log-SD, minimum, maximum, expected covered bases) of the clipped lognormal read-length law."""
    mean, sigma = float(model.long_read_mean_bp), float(model.long_read_length_sigma)
    mu = math.log(mean) - sigma * sigma / 2
    f_min = LONG_READ_MIN_BP
    f_max = max(f_min + 1, int(math.ceil(math.exp(mu + LONG_READ_SIGMA_LIMIT * sigma))) + 1)
    quantiles = (np.arange(FRAGMENT_QUANTILES) + 0.5) / FRAGMENT_QUANTILES
    lengths = np.clip(np.rint(np.exp(mu + sigma * ndtri(quantiles))), f_min, f_max)
    return mu, sigma, f_min, f_max, float(np.mean(lengths))


def _window_alt_counts(haplotype, positions, window):
    """ALT alleles of one homologue within +-window bp of each site (the site included)."""
    lo = np.searchsorted(positions, positions - window)
    hi = np.searchsorted(positions, positions + window, side="right")
    cumulative = np.r_[0, np.cumsum(haplotype, dtype=np.int64)]
    return cumulative[hi] - cumulative[lo]


def _long_read_sample(hap0, hap1, positions, depth, model, factor, site_factor, rng, context, present=None,
                      cohort=None, *, batch_key, contamination=None):
    """REF/ALT counts int64[L, 2] of one sample under the long-read sampler (module docstring), from its own stream
    (later read batches from their own: module docstring, Batches).

    present: bool [2, L] (structure) or None; a homologue's hits at its absent cells are removed. cohort: (hap0 [N, L],
    hap1 [N, L], this sample's row) for foreign reads. batch_key: (seed, sample index) of the batch streams."""
    sites = len(hap0)
    mu, sigma, f_min, f_max, covered_per_read = _long_read_lengths(model)
    width = int(sites * model.dropout_fraction)
    lo = int(positions[0]) - f_max
    span = int(positions[-1]) + 1 - lo
    counts = np.zeros(sites, dtype=np.int64)
    alt = np.zeros(sites, dtype=np.int64)
    rate = depth * factor / (1 - width / sites) / covered_per_read
    target = site_factor
    if model.count_dispersion:
        target = rng.gamma(1 / model.count_dispersion, model.count_dispersion, size=sites)
        if site_factor is not None:
            target *= site_factor
    if target is None:
        n = int(rng.poisson(rate * span))
        windows = {}
    else:
        edges, widths, local = _fragment_envelopes(positions, target, lo, span, f_max)
        per_window = rng.poisson(rate * local * widths)
        n = int(per_window.sum())
        windows = dict(edges=edges, widths=widths, local=local, window_ends=np.cumsum(per_window))
    window_alt = library_alt = bounds = None
    if model.reference_bias:
        window = int(model.reference_bias_window_bp)
        window_alt = (_window_alt_counts(hap0, positions, window), _window_alt_counts(hap1, positions, window))
        # A donor read's local loss counts the donor homologue's ALT alleles: the library's donor once per library,
        # a uniform foreign read's donor per batch (prefix sums over the same windows).
        bounds = (np.searchsorted(positions, positions - window),
                  np.searchsorted(positions, positions + window, side="right"))
        if contamination is not None:
            library_alt = (_window_alt_counts(contamination[2], positions, window),
                           _window_alt_counts(contamination[3], positions, window))
    size_b = _batch_size(covered_per_read, sites, span)
    for g, m, start, envelope in _batches(rng, n, size_b, batch_key, **windows):
        if start is None:
            start = lo + g.integers(0, span, size=m)
        length = np.clip(np.rint(np.exp(g.normal(mu, sigma, size=m))), f_min, f_max).astype(np.int64)
        which = g.integers(0, 2, size=m)
        quality = np.clip(g.normal(model.read_quality_mean, model.read_quality_sd, size=m), READ_QUALITY_MIN,
                          READ_QUALITY_MAX)
        error = 10.0 ** (-quality / 10)
        # A foreign read takes a uniform other sample's homologue (which), or the contaminated library's donor;
        # drawn only when a share is positive (_read_donors).
        donor = _read_donors(g, m, which, model, cohort, contamination)
        mismapped = None
        if model.chimera_share:
            start, length, which, error, donor, envelope, mismapped = _chimera_split(
                g, start, length, which, error, donor, envelope, model, lo, span, windows)
            m = len(start)
        first = np.searchsorted(positions, start)
        stop = np.searchsorted(positions, start + length)
        size = stop - first
        keep = size > 0
        if not np.any(keep):
            continue
        first, size = first[keep], size[keep]
        reads = np.flatnonzero(keep)
        offsets = np.arange(int(size.sum())) - np.repeat(np.cumsum(size) - size, size)
        site = np.repeat(first, size) + offsets
        del offsets
        read = np.repeat(reads, size)
        if target is not None:
            mark = g.random(m)
            retained = mark[read] < target[site] / envelope[read]
            site, read = site[retained], read[retained]
            del retained
        owner = which[read]
        if present is not None:
            here = present[owner, site] if donor is None else present[owner, site] | (donor[read] != OWN)
            site, read, owner = site[here], read[here], owner[here]
            del here
        allele = np.where(owner == 0, hap0[site], hap1[site]).astype(np.int64)
        foreign = None
        if donor is not None:
            foreign = donor[read] != OWN
            allele[foreign] = _donor_alleles(donor[read][foreign], site[foreign], owner[foreign], cohort,
                                             contamination)[0]
        if mismapped is not None and mismapped.any():
            # A mismapped chimeric segment reports its own locus' alleles at the partner's sites: unrelated to the
            # local genotype, drawn as its source homologue's allele (its own, or its donor's) at a uniform other site.
            moved = mismapped[read]
            other = g.integers(0, sites, size=int(moved.sum()))
            allele[moved] = np.where(owner[moved] == 0, hap0[other], hap1[other])
            if foreign is not None and (moved & foreign).any():
                both = moved & foreign
                allele[both] = _donor_alleles(donor[read][both], other[foreign[moved]], owner[both], cohort,
                                              contamination)[0]
        donor_hits = None if foreign is None else (donor[read][foreign], foreign)
        e = error[read]
        del read
        substitution = model.substitution_share
        lost = 1 - (1 - e * (1 - substitution) / 2) * (1 - e * substitution * 2 / 3)
        other = e * substitution / 3
        if model.reference_pull:
            pull = np.minimum(1.0, model.reference_pull * e / _mean_error(model))
            other = np.where(allele == 1, 1 - (1 - other) * (1 - pull), other)
        if window_alt is not None:
            k = np.where(owner == 0, window_alt[0][site], window_alt[1][site])
            if donor_hits is not None and donor_hits[1].any():
                k = k.copy()
                k[donor_hits[1]] = _donor_window_alt(donor_hits[0], site[donor_hits[1]], owner[donor_hits[1]],
                                                     cohort, library_alt, bounds)
            lost = 1 - (1 - lost) * np.exp(-model.reference_bias * k.astype(np.float64) ** model.reference_bias_power)
        if context is not None:
            extending, strength = context
            at = (extending[site] >= 0) & (allele == extending[site])
            indel = np.where(at, np.minimum(1.0, e * (1 - substitution) * model.context_factor * strength[site]), 0.0)
            lost = 1 - (1 - lost) * (1 - indel * (1 - model.context_convert))
            other = 1 - (1 - other) * (1 - indel * model.context_convert)
        u = g.random(len(site))
        dropped = u < lost
        flipped = ~dropped & (u < lost + (1 - lost) * other)
        allele[flipped] ^= 1
        site, allele = site[~dropped], allele[~dropped]
        counts += np.bincount(site, minlength=sites).astype(np.int64)
        alt += np.bincount(site, weights=allele, minlength=sites).astype(np.int64)
    if width:
        cut = int(rng.integers(0, sites - width + 1))
        counts[cut : cut + width] = 0
        alt[cut : cut + width] = 0
    return np.stack((counts - alt, alt), axis=-1).astype(np.int64, copy=False)
