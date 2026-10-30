"""Relatedness stage: pairwise identity by descent and full-sib families from genotype likelihoods alone.

Entry point ``run(store, contigs, sample_ids, config, genetic_maps, budget) -> RelatednessProduct``; the pedigree
release reads it (the sibling-as-parent guard). It runs after evidence and uses no painting, pedigree, phase or
cohort label, so no later stage feeds back into it.
"""
from .config import RelatednessConfig
from .records import RelatednessProduct
from .stage import run

__all__ = ["RelatednessConfig", "RelatednessProduct", "run"]
