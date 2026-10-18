"""Evidence stage: allele depths of one contig -> site/sample QC -> read model -> genotype likelihoods.

Public entry points: :func:`prepare_contig` (pure), :func:`run_contig` / :func:`run` (store),
:func:`canonical_order_source` (the run's canonical sample order, applied by the driver to its source); types in
``haplotype_reconstruction.model.evidence``; settings in :mod:`.config`; the record in :mod:`.records`.
"""
from .config import (DiscoverySiteRule, EvidenceConfig, ReadModelConfig, SampleQualityConfig,
                     SiteQualityConfig)
from .reports import EvidenceReport, EvidenceSummary
from .sample_order import CanonicalOrderSource, canonical_order_source
from .stage import PreparedEvidence, prepare_contig, run, run_contig

__all__ = ["CanonicalOrderSource", "DiscoverySiteRule", "EvidenceConfig", "EvidenceReport", "EvidenceSummary",
           "PreparedEvidence", "ReadModelConfig", "SampleQualityConfig", "SiteQualityConfig",
           "canonical_order_source", "prepare_contig", "run", "run_contig"]
