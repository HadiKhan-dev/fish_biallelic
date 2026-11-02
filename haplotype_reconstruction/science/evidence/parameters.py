"""Declare unchanged evidence parameters with their 3.5 provenance (EC01-24)."""

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
        default=5, metadata={"provenance": "B/evidence/config.py: minimum_total_depth; P0 evidence-core.parameters.csv"}
    )

    def threshold(self) -> int:
        """The unchanged discovery support floor, independent of cohort size."""
        return self.minimum_total_depth


@dataclass(frozen=True)
class EvidenceParameters:
    """Evidence stage: loader policy -> site/sample QC -> read model -> likelihoods -> blocks."""

    # Discovery block width in sites (consecutive, non-overlapping windows of the site axis).
    block_sites: int = field(
        default=200, metadata={"provenance": "B/evidence/config.py: block_sites; P0 evidence-core.parameters.csv"}
    )
    discovery_sites: DiscoverySiteRule = field(
        default_factory=DiscoverySiteRule,
        metadata={"provenance": "B/evidence/config.py: discovery_sites; P0 evidence-core.parameters.csv"},
    )
    site_quality: SiteQualityParameters = field(
        default_factory=SiteQualityParameters,
        metadata={"provenance": "B/evidence/config.py: site_quality; P0 evidence-core.parameters.csv"},
    )
    sample_quality: SampleQualityParameters = field(
        default_factory=SampleQualityParameters,
        metadata={"provenance": "B/evidence/config.py: sample_quality; P0 evidence-core.parameters.csv"},
    )
    read_model: ReadModelParameters = field(
        default_factory=ReadModelParameters,
        metadata={"provenance": "B/evidence/config.py: read_model; P0 evidence-core.parameters.csv"},
    )


# Provenance: B/evidence/stage.py:29, EC23; absolute probability validation tolerance.
ROW_SUM_TOLERANCE = 1e-12

__all__ = ["DiscoverySiteRule", "EvidenceParameters"]
