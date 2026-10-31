"""Settings of the evidence stage. Defaults are the production values of base (92b3daf).

User-visible (TOML ``[quality]``, mapped by ``config/stages.py``): ``site_quality.enabled``
(``site_filter``: on for VCF input, off for simulations by default), ``sample_quality.enabled``
(``sample_filter``; it runs only where the site filter is applied) and ``read_model.enabled``
(``read_calibration``). Every other field is an internal model constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiscoverySiteRule:
    """Minimum read support for discovery, independent of cohort size.

    Discovery and local selection fit and release alleles at sites with at least
    ``minimum_total_depth`` reads across all analysed samples. Other sites remain
    on the site axis for later filling and inference. This is a read-support floor,
    not a requirement that a fixed fraction of the cohort observes the site.
    """

    minimum_total_depth: int = 5

    def threshold(self, n_samples: int) -> float:
        """Keep the stage's call signature; cohort size does not change support."""
        return self.minimum_total_depth


@dataclass(frozen=True)
class SiteQualityConfig:
    """Self-calibrated site filter r1-r5 (``evidence/site_quality.py``).

    Every removal threshold comes from the contig's own distributions: an empirical null
    (histogram mode and half-maximum widths) for total depth (r1) and per-sample depth
    heterogeneity (r2), clean-core/artefact mixtures for wrong-allele reads in homozygotes (r3)
    and heterozygote allele balance (r4). These fields fix only the genotype-classification read
    model, the posterior levels and the identifiability guard. Rule r0 (not a biallelic SNP) is
    applied by the loader for every input.
    """

    enabled: bool = True
    # Read model used only to classify each sample's genotype at a site; the clean-core error
    # rate and allele balance are then fitted, not these.
    classification_error: float = 0.01
    classification_heterozygote_alt: float = 0.5
    # r1/r2 windows: where the fitted core's local posterior is >= this level (histogram
    # smoothed over this many bins).
    core_posterior_level: float = 0.5
    histogram_smoothing_bins: int = 5
    # r3/r4: a site is flagged when P(clean core) < this level, and only if the artefact class is
    # separated from the clean core by this factor (r3: rate ratio; r4: latent standard deviation or
    # mean offset over max(clean latent standard deviation, binomial standard deviation at the
    # median depth)).
    clean_posterior_level: float = 0.5
    minimum_artefact_separation: float = 3.0
    # r5 (informativeness, not quality): fewer expected carriers of a non-major genotype than
    # this. It removes sites without segregating information.
    drop_uninformative_sites: bool = True
    minimum_expected_carriers: float = 2.0
    # Contigs with fewer biallelic SNPs than this are not filtered.
    minimum_calibration_sites: int = 1000


@dataclass(frozen=True)
class SampleQualityConfig:
    """Per-sample, per-contig quality exclusion from founder construction (``evidence/sample_quality.py``).

    Runs only where the site filter is applied (it uses the r1-r4-passing SNPs, r3's clean error rate and r4's clean
    heterozygote balance h0, widened to r4's two component means when r4 records an unresolved fit). Per library, a
    mixture over its eligible cells (no other library carries the minor allele) tests for a foreign-like minor-read
    component above ``minimum_foreign_fraction`` / 2, with a common share within the non-artefact mixture, against the
    library's own errors (any distribution below that amplitude), its own heterozygotes (beta-binomial, the same in
    both orientations) and local artefacts (a free share per region, each with its own fraction); the library is
    flagged on a contig when the statistic reaches the working threshold. A flag is a quality exclusion, not a
    contamination diagnosis: elevated library error, dispersed variable-fraction artefacts and biased private alleles
    can flag too, even in a short tract if other regions have little eligible information. Flagged samples leave founder construction on that contig
    (``ContigEvidence.founder_view``) and the r5 carrier count. Heterozygous near-private alleles (inherited segments)
    are reported.
    """

    enabled: bool = True
    # q: a library carries the cohort-minor allele when P(K >= k | d, e) <= q / (N - 1) (a site nobody carries
    # shows a carrier anywhere with probability <= q). The same q sets the working threshold of the per-library test:
    # T >= u with P(chi2_1 >= u) = 2 q / (G N) (the boundary law (chi2_0 + chi2_1) / 2 of one foreign fraction, by
    # Bonferroni over the G fractions and the N libraries). That level is nominal only for a regular, correctly
    # specified likelihood of independent windows; the measured clean false-flag rate is what supports it.
    carrier_tail_level: float = 0.05
    # alpha_min: twice the working foreign-allele amplitude a_min. The null admits any error distribution of the
    # library up to a_min = alpha_min / 2; a flag needs a component above it. Motivation (not a derived safety
    # boundary): the founder-construction and painting emissions mix each observed site's genotype evidence with a
    # uniform genotype error of mass r = 0.01 (assembly LinkerConfig.robustness, PaintingConfig.robustness_epsilon,
    # the founder refit), which limits the influence of discordant observations. At a site where foreign DNA at read
    # fraction alpha is homozygous for an allele the sample lacks, a cell of depth d shows K ~ Bin(d, alpha) foreign
    # reads; two or more contradict a homozygote decisively (heterozygote likelihood ratio above 3 (1 - r) / r at
    # clean error rates 1e-4..1e-3 and depths of about 4 to 16) with probability ~ C(d, 2) alpha^2, which is of the
    # order of r at alpha = 0.8-1.4% over 11-18x (alpha 1%: 0.72% at 13x; alpha 2%: 2.7%). A foreign allele shows at
    # alpha (foreign homozygote) or alpha/2 (heterozygote), so a = alpha_min / 2 aims to catch 1% contamination of
    # either zygosity; a flag does not establish alpha >= 1%, and sensitivity near the amplitude is measured, not
    # guaranteed. Libraries whose cells are mostly at 1-5x are more fragile; the QC has little power there. Not
    # calibrated on any real library.
    minimum_foreign_fraction: float = 0.01
    # f: foreign alleles are resolved up to a_hi = f h_lo / 2 (h0/4: a foreign heterozygote at alpha/2 with alpha up
    # to h0/2), with h_lo the lower end of the balance interval (h0, or the lower of r4's two component means when r4's
    # labelling is unresolved); grosser contamination is detected at that ceiling, its fraction reported as at least
    # it. Own private heterozygotes are beta-binomial about the balance interval with a latent SD up to 0.1, the same
    # for a private ALT and a private REF allele. Own private alleles shown below about h_lo - 0.2 (reference or
    # haplotype bias) can resemble foreign alleles, even locally if other regions carry little eligible information.
    # The fitted fraction is not a calibrated contamination estimate.
    heterozygote_floor_ratio: float = 0.5
    # Own private heterozygotes and errors may be commoner at some depths (mapping bias lowers depth and allele
    # fraction together): their weights are fitted in this many depth classes (the library's eligible-cell depth
    # quantiles), foreign alleles' share in none (foreign reads do not depend on a site's depth).
    heterozygote_depth_classes: int = 4
    # Kept from release 2.16: smaller cohorts flag nothing (with few libraries most alleles absent from every other
    # library are the library's own, and the eligible set is too small to separate foreign alleles from them).
    minimum_samples: int = 20
    # Local artefacts (both hypotheses): the contig's passing sites are cut into this many regions of equal site
    # counts, and in each region a free share of the library's cells may be local artefacts (a collapsed repeat, a
    # mismapped paralog, a private tract with mapping bias), each with a minor fraction of its own (uniform on [0, 1],
    # so k is uniform on 0..d). The foreign share c is common within the non-artefact mixture; its marginal weight
    # is (1 - pi_R) c in region R. No independent regional support is required; local excesses can be absorbed,
    # but need not be. The uniform law and region count are working choices. 1: one contig-wide artefact share.
    artefact_regions: int = 16
    # Cells of one sample closer than one sequenced fragment can share reads: each eligible cell weighs 1 / (the
    # library's eligible cells in its fragment_bp window, counted from the first passing site), and segment-report
    # cells form one event.
    fragment_bp: int = 1000
    # Inherited-segment report only: heterozygous near-private events of one (sample, partner)
    # less than segment_gap_bp apart form a region; regions with fewer events are not listed.
    segment_gap_bp: int = 1_000_000
    segment_minimum_events: float = 3.0


@dataclass(frozen=True)
class ReadModelConfig:
    """Held-out calibration of the observed-AD read model (``evidence/read_model``).

    Fitting uses every kept site, cells of depth 1..``maximum_fit_depth``, and four physical folds of
    ``fold_window_bp`` windows (folds 0/1 fit, folds 2/3 are held out). The candidates are nested:
    shared binomial (EM from ``em_starts``), pooled sample binomial and sample beta-binomial
    heterozygotes (joint concentration); homozygotes are binomial. A more complex candidate is
    selected when its held-out gain is at least ``selection_z`` standard errors over the held-out
    stretches. When disabled, or when the shared model has no finite held-out score, the fixed
    binomial model (``EvidenceConfig.fixed_error``) applies.
    """

    enabled: bool = True
    # Selection: a more complex candidate is accepted when its held-out log-likelihood gain is at least
    # selection_z standard errors. The error comes from the gains of the held-out stretches (folds 2/3 of each
    # four-window cycle), since sites linked through ancestry or sharing reads are not independent observations.
    selection_z: float = 2.0
    # With fewer held-out stretches than this the error cannot be estimated and the shared model is kept.
    selection_minimum_blocks: int = 4
    maximum_fit_depth: int = 48
    fold_window_bp: int = 1_000_000
    em_starts: tuple[tuple[float, float], ...] = ((0.02, 0.5), (0.01, 0.35), (0.08, 0.65))
    em_maximum_iterations: int = 2000
    em_tolerance_per_observation: float = 1e-10
    # Start of the joint fit's shared heterozygote concentration (kappa).
    initial_heterozygote_concentration: float = 40.0
    # Likelihood application: cells up to this depth come from a per-sample lookup table,
    # deeper cells are evaluated directly (a performance setting, not a depth filter).
    lookup_depth: int = 64


@dataclass(frozen=True)
class EvidenceConfig:
    """Evidence stage: loader policy -> site/sample QC -> read model -> likelihoods -> blocks."""

    # Discovery block width in sites (consecutive, non-overlapping windows of the site axis).
    block_sites: int = 200
    discovery_sites: DiscoverySiteRule = field(default_factory=DiscoverySiteRule)
    site_quality: SiteQualityConfig = field(default_factory=SiteQualityConfig)
    sample_quality: SampleQualityConfig = field(default_factory=SampleQualityConfig)
    read_model: ReadModelConfig = field(default_factory=ReadModelConfig)
    # Fallback read model: binomial with this symmetric error and heterozygote ALT fraction 1/2.
    fixed_error: float = 0.02
