"""Read-support, quality, calibration and blocking controls for evidence."""

from __future__ import annotations

from dataclasses import dataclass, field

from .quality.parameters import SampleQualityParameters, SiteQualityParameters
from .read_model.parameters import ReadModelParameters


@dataclass(frozen=True)
class DiscoverySiteRule:
    """Minimum read support for discovery, independent of cohort size.

    Discovery and local selection fit and release alleles at sites with at least
    ``minimum_total_depth`` reads across all analysed samples. Other sites remain
    on the site axis for later filling and inference. This is a read-support floor,
    not a requirement that a fixed fraction of the cohort observes the site.
    """

    minimum_total_depth: int = field(
        default=5,
        metadata={
            "provenance": (
                "Minimum total reads across analysed samples at a discovery site (reads); keeps "
                "sites with insufficient support out of allele fitting."
            ),
        },
    )

    def threshold(self) -> int:
        """The discovery support floor, independent of cohort size."""
        return self.minimum_total_depth


@dataclass(frozen=True)
class EvidenceParameters:
    """Evidence stage: loader policy -> site/sample QC -> read model -> likelihoods -> blocks."""

    # Discovery block width in sites (consecutive, non-overlapping windows of the site axis).
    block_sites: int = field(
        default=200,
        metadata={
            "provenance": (
                "Consecutive kept sites per discovery block (sites); fixes the non-overlapping "
                "discovery windows."
            ),
        },
    )
    discovery_sites: DiscoverySiteRule = field(
        default_factory=DiscoverySiteRule,
        metadata={
            "provenance": (
                "Read-support rule for discovery (parameter group); retains low-support sites on "
                "the evidence axis for later inference."
            ),
        },
    )
    site_quality: SiteQualityParameters = field(
        default_factory=SiteQualityParameters,
        metadata={
            "provenance": (
                "Contig-calibrated site-filter controls (parameter group); sets classification, "
                "posterior and identifiability rules."
            ),
        },
    )
    sample_quality: SampleQualityParameters = field(
        default_factory=SampleQualityParameters,
        metadata={
            "provenance": (
                "Per-library founder-exclusion controls (parameter group); separates foreign-like "
                "read evidence from own alleles and local artefacts."
            ),
        },
    )
    read_model: ReadModelParameters = field(
        default_factory=ReadModelParameters,
        metadata={
            "provenance": (
                "Observed-allele-depth read-model controls (parameter group); governs held-out "
                "calibration and likelihood application."
            ),
        },
    )


# Absolute normalized genotype-probability row-sum tolerance (probability).
ROW_SUM_TOLERANCE = 1e-12

__all__ = ["DiscoverySiteRule", "EvidenceParameters"]
