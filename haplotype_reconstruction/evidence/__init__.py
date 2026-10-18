"""Evidence stage: allele depths of one contig -> site/sample QC -> read model -> genotype likelihoods.

Public entry points: :func:`prepare_contig` (pure), :func:`run_contig` / :func:`run` (store); types in
``haplotype_reconstruction.model.evidence``; settings in :mod:`.config`; the record in :mod:`.records`.
"""
from .config import (DiscoverySiteRule, EvidenceConfig, ReadModelConfig, SampleQualityConfig,
                     SiteQualityConfig)
from .reports import EvidenceReport, EvidenceSummary
from .stage import PreparedEvidence, prepare_contig, run, run_contig

__all__ = ["DiscoverySiteRule", "EvidenceConfig", "EvidenceReport", "EvidenceSummary", "PreparedEvidence",
           "ReadModelConfig", "SampleQualityConfig", "SiteQualityConfig", "prepare_contig", "run",
           "run_contig"]
