"""Pedigree stage: observed-parent states and identities from paintings and raw genotype likelihoods.

Entry point ``run(store, axes, eligibility, config, budget) -> PedigreeProduct``; family phase and
recombination read ``PedigreeProduct.edges`` (ParentEdges), evaluation reads ``.calls``.
"""
from .config import PedigreeConfig
from .eligibility import ParentEligibility
from .records import ParentEdges, PedigreeCalls, PedigreeProduct
from .stage import run, tier_b_minimum_contigs

__all__ = ["PedigreeConfig", "ParentEligibility", "ParentEdges", "PedigreeCalls",
           "PedigreeProduct", "run", "tier_b_minimum_contigs"]
