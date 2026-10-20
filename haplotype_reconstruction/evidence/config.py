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
    # separated from the clean core by this factor (rate ratio, or binomial standard deviations
    # at the median depth).
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
    """Per-sample, per-contig contamination QC (``evidence/sample_quality.py``).

    Runs only where the site filter is applied (it uses the r1-r4-passing SNPs and r4's clean
    heterozygote balance). A sample is flagged on a contig when the robust z of its
    low-fraction private-allele statistic exceeds ``flag_robust_z``; flagged samples leave
    founder construction on that contig (``ContigEvidence.founder_view``) and the r5 carrier
    count. Heterozygous near-private alleles (inherited segments) never flag; they are reported.
    """

    enabled: bool = True
    # Extreme-outlier bar on the variance-stabilised scale (real TroMau, 22 contigs: samples
    # without a contamination signature reach at most z 4.0).
    flag_robust_z: float = 5.0
    # The median/MAD spread needs a cohort; smaller cohorts flag nothing.
    minimum_samples: int = 20
    # Cells of one sample closer than one sequenced fragment can share reads: one event.
    fragment_bp: int = 1000
    # Inherited-segment report only: heterozygous near-private events of one (sample, partner)
    # less than segment_gap_bp apart form a region; regions with fewer events are not listed.
    segment_gap_bp: int = 1_000_000
    segment_minimum_events: float = 3.0


@dataclass(frozen=True)
class ReadModelConfig:
    """Held-out calibration of the observed-AD read model (``evidence/read_model``).

    Fitting uses every ``marker_stride``-th site, cells of depth 1..``maximum_fit_depth``, and
    four physical folds of ``fold_window_bp`` windows (folds 0/1 fit, fold 2 selects, fold 3 is
    diagnostic). The candidates are nested: shared binomial (EM from ``em_starts``), pooled
    sample binomial and sample beta-binomial heterozygotes (joint concentration); homozygotes are
    binomial. When disabled, or when no candidate has a finite selection score, the fixed binomial
    model (``EvidenceConfig.fixed_error``) applies.
    """

    enabled: bool = True
    marker_stride: int = 8
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
