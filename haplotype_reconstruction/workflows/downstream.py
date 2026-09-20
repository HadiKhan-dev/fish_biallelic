"""One-way painting → pedigree → family phase → recombination orchestration.

Dataset adapters own sample selection and eligibility. This module deliberately
has no dataset names, pedigree truth, or feedback into upstream reconstruction.
"""
from __future__ import annotations

from ..pedigree import pipeline as pedigree
from ..refinement import pipeline as refinement
from ..refinement.model import FamilyRefinementConfig
from ..recombination import pipeline as recombination
from ..recombination.model import RecombinationMapConfig


def run_downstream(
    store, contigs, sample_ids, *, output_dir, n_workers,
    raw_gl_stage, raw_sites_stage, raw_gl_key="global_probs",
    parent_eligibility=None, genetic_maps=None, recombination_rate=5e-8,
    all_contigs=None, publish_global=True,
):
    """Run pedigree–recombination with one observation source and one shared CPU ceiling.

    pedigree may prepare individual contigs for a distributed run. Such a shard
    cannot publish a genome-wide pedigree or run family-dependent stages;
    the coordinator calls again with all required chromosomes available.
    """
    _, payload = pedigree.run_pedigree(
        store, contigs, sample_ids,
        output_dir=output_dir, n_workers=n_workers,
        raw_gl_stage=raw_gl_stage, raw_sites_stage=raw_sites_stage,
        raw_gl_key=raw_gl_key, parent_eligibility=parent_eligibility,
        all_contigs=all_contigs, publish_global=publish_global,
        genetic_maps=genetic_maps, recombination_rate=recombination_rate,
    )
    if not publish_global:
        return payload
    final_contigs = contigs if all_contigs is None else all_contigs
    print("FAMILY PHASE: pedigree-conditioned refinement and final phase polishing")
    refinement.run_refinement(
        store, final_contigs, sample_ids,
        pedigree_payload=payload, output_dir=output_dir,
        raw_gl_stage=raw_gl_stage, raw_sites_stage=raw_sites_stage,
        raw_gl_key=raw_gl_key, n_workers=n_workers,
        genetic_maps=genetic_maps,
        config=FamilyRefinementConfig(recombination_rate=recombination_rate),
    )
    recombination.run_recombination(
        store, final_contigs, sample_ids,
        pedigree_payload=payload, output_dir=output_dir, n_workers=n_workers,
        genetic_maps=genetic_maps,
        config=RecombinationMapConfig(recombination_rate=recombination_rate),
    )
    return payload
