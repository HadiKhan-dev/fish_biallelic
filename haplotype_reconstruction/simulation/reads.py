"""Sequencing reads of simulated individuals (the generating read model, never the inference one).

Two samplers, chosen by ``ReadModel.read_length_bp``:

* independent sites (0): per cell, depth ~ Poisson(depth x sample factor / (1 - dropout share))
  and ALT reads ~ Binomial(depth, p_g), p = (error_rate, heterozygote_alt_probability,
  1 - error_rate); a lognormal sample factor (mean one, coefficient of variation depth_cv) and one
  contiguous dropout tract of dropout_fraction of the sites per sample are drawn only when
  configured;
* paired-end fragments (> 0): per sample, fragment starts form a Poisson process along the contig;
  a fragment has insert length F ~ Normal(fragment_mean_bp, fragment_sd_bp) rounded and clipped to
  [50 bp, mean + 6 sd], comes from one of the sample's two haplotypes with probability 1/2 and is
  read over read_length_bp at each end (bases of overlapping mates counted once). The start rate is
  depth / E[bases covered per fragment], so every site keeps the Poisson(depth) marginal while
  neighbouring sites become dependent. depth_cv and dropout_fraction act as above. An allelic bias
  p = heterozygote_alt_probability != 1/2 is applied per read at heterozygous sites (a read of the
  disfavoured allele reports the other one with probability |2p - 1|); the error rate acts on
  homozygous cells only.

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
  combine as independent hazards. A share foreign_read_share of reads comes from a uniform other sample's homologue.
  Context sites come from the contig stream after every draw of the other samplers, so those are unchanged.

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

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri

from ..core.errors import SettingsError
from ..core.random import stream
from .design import ReadModel
from .parameters import (
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
    return ReadInputs(hap0, hap1, positions, factor, site_factor, float(depth), model, seed, context, strength)


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
    for row, i in enumerate(range(first, stop)):
        rng = stream("simulation.sample_reads", seed=inputs.seed, index=i)
        args = (inputs.hap0[i], inputs.hap1[i], inputs.positions, inputs.depth, inputs.model, float(inputs.factor[i]))
        target = inputs.site_factor
        if inputs.model.region_depth_cv:
            # The regional factors have their own stream per sample, so the reads are sampled once and a
            # zero regional CV leaves the sample's read stream, and so every count, as it was.
            regional = _region_factors(
                inputs.positions, inputs.model, stream("simulation.sample_region", seed=inputs.seed, index=i)
            )
            target = regional if target is None else target * regional
        structure = inputs.structure
        if model.long_read_mean_bp:
            present = None if structure is None else structure.present[i]
            context = None if inputs.context is None else (inputs.context, inputs.context_strength)
            counts = sample(*args, target, rng, context, present, (inputs.hap0, inputs.hap1, i), batch_key=(inputs.seed, i))
        elif model.read_length_bp:
            counts = sample(*args, target, rng, batch_key=(inputs.seed, i))
        else:
            counts = sample(*args, target, rng)
        if structure is not None and (not structure.present[i].all() or (structure.copies[i] >= 0).any()):
            # Expected reads of one homologue copy (unimpaired depth x sample, site and regional factors).
            lam = np.full(sites, inputs.depth * float(inputs.factor[i]) / 2)
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
                thin=not model.long_read_mean_bp,
            )
        if model.uncalled_base_probability:
            counts -= rng.binomial(counts, model.uncalled_base_probability)
        out[row] = counts
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
    """REF/ALT counts int64[L, 2] of one sample under the independent-site sampler, from its own stream."""
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


def _fragment_sample(hap0, hap1, positions, depth, model, factor, site_factor, rng, *, batch_key):
    """REF/ALT counts int64[L, 2] of one sample under the paired-end fragment sampler, from its own stream (later
    read batches from their own: module docstring, Batches)."""
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
    bias = model.heterozygote_alt_probability - 0.5
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
    size_b = _batch_size(covered_per_fragment, sites, span)
    for g, m, start, envelope in _batches(rng, n, size_b, batch_key, **windows):
        if start is None:
            start = lo + g.integers(0, span, size=m)
        length = np.clip(np.rint(g.normal(mean, sd, size=m)), f_min, f_max).astype(np.int64)
        which = g.integers(0, 2, size=m)
        first_end = start + np.minimum(length, r)
        second_start = np.maximum(start + length - r, first_end)
        second_end = start + length
        intervals = (
            (np.searchsorted(positions, start), np.searchsorted(positions, first_end)),
            (np.searchsorted(positions, second_start), np.searchsorted(positions, second_end)),
        )
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
            if target is not None:
                fragment_parts.append(np.repeat(np.flatnonzero(keep), size))
        if not site_parts:
            continue
        site = np.concatenate(site_parts)
        owner = np.concatenate(hap_parts)
        del site_parts, hap_parts
        if target is not None:
            # One uniform mark per fragment couples retention across all its hits.
            # Shared-hit intensity is proportional to min(target_i, target_j), so an
            # unrelated high-depth site cannot change local dependence via the envelope.
            mark = g.random(m)
            fragment = np.concatenate(fragment_parts)
            keep = mark[fragment] < target[site] / envelope[fragment]
            site, owner = site[keep], owner[keep]
            del fragment, keep
        allele = np.where(owner == 0, hap0[site], hap1[site]).astype(np.int64)
        het = hap0[site] != hap1[site]
        if bias:
            favoured = 1 if bias > 0 else 0
            convert = het & (allele != favoured) & (g.random(len(site)) < 2 * abs(bias))
            allele[convert] = favoured
        allele ^= (~het) & (g.random(len(site)) < model.error_rate)
        counts += np.bincount(site, minlength=sites).astype(np.int64)
        alt += np.bincount(site, weights=allele, minlength=sites).astype(np.int64)
    if width:
        first = int(rng.integers(0, sites - width + 1))
        counts[first : first + width] = 0
        alt[first : first + width] = 0
    return np.stack((counts - alt, alt), axis=-1).astype(np.int64, copy=False)


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
                      cohort=None, *, batch_key):
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
    bias = model.heterozygote_alt_probability - 0.5
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
    window_alt = None
    if model.reference_bias:
        window = int(model.reference_bias_window_bp)
        window_alt = (_window_alt_counts(hap0, positions, window), _window_alt_counts(hap1, positions, window))
    size_b = _batch_size(covered_per_read, sites, span)
    for g, m, start, envelope in _batches(rng, n, size_b, batch_key, **windows):
        if start is None:
            start = lo + g.integers(0, span, size=m)
        length = np.clip(np.rint(np.exp(g.normal(mu, sigma, size=m))), f_min, f_max).astype(np.int64)
        which = g.integers(0, 2, size=m)
        quality = np.clip(g.normal(model.read_quality_mean, model.read_quality_sd, size=m), READ_QUALITY_MIN,
                          READ_QUALITY_MAX)
        error = 10.0 ** (-quality / 10)
        donor = None
        if model.foreign_read_share and cohort is not None and len(cohort[0]) > 1:
            # A foreign read takes a uniform other sample's homologue (which); drawn only when the share is positive.
            foreign = g.random(m) < model.foreign_read_share
            other = g.integers(0, len(cohort[0]) - 1, size=m)
            donor = np.where(foreign, other + (other >= cohort[2]), -1)
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
            here = present[owner, site] | (donor is not None and donor[read] >= 0)
            site, read, owner = site[here], read[here], owner[here]
            del here
        allele = np.where(owner == 0, hap0[site], hap1[site]).astype(np.int64)
        het = hap0[site] != hap1[site]
        if donor is not None:
            outside = donor[read] >= 0
            d, s_, o = donor[read][outside], site[outside], owner[outside]
            allele[outside] = np.where(o == 0, cohort[0][d, s_], cohort[1][d, s_])
            het = het & ~outside
        if bias:
            favoured = 1 if bias > 0 else 0
            convert = het & (allele != favoured) & (g.random(len(site)) < 2 * abs(bias))
            allele[convert] = favoured
        del het
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
