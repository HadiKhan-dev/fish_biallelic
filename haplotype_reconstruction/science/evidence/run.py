"""Prepare a contig: site/sample quality, read-model calibration, likelihoods and discovery blocks."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from haplotype_reconstruction.contracts.evidence import ContigEvidence, EvidenceDecisions
from haplotype_reconstruction.contracts.likelihoods import CodedLikelihoods
from haplotype_reconstruction.contracts.observations import PairArray, pair_views
from haplotype_reconstruction.contracts.roles import record_field
from haplotype_reconstruction.contracts.source import ContigReads
from haplotype_reconstruction.core.concurrency import ThreadSource
from haplotype_reconstruction.core.errors import InputError

from ..shared.cohort import cohort_genotype_frequencies
from .blocks import _block_spans
from .compact import build_likelihoods
from .likelihoods import _observation_mask
from .likelihoods import likelihoods_by_sample as likelihoods_by_sample
from .parameters import EvidenceParameters
from .quality.site_quality import filter_sites
from .read_sharing import ReadSharing
from .read_model import calibrate as calibration
from .read_model.fit import fixed_model


@dataclass(frozen=True)
class PreparedEvidence:
    evidence: ContigEvidence = record_field("state", dtype="ContigEvidence")
    decisions: EvidenceDecisions = record_field("state", dtype="EvidenceDecisions")
    read_sharing: ReadSharing = record_field("state", dtype="ReadSharing")


def prepare_contig(
    reads: ContigReads, config: EvidenceParameters, team: ThreadSource = 1, *, map_work=map, reverse: bool = False
) -> PreparedEvidence:
    """The evidence of one contig from its allele depths.

    Order (as base): site filter and contamination QC on all input SNPs; read-model calibration
    on the kept sites of every analysed sample (flagged samples included); normalised
    likelihoods; observation mask; discovery sites (all samples); 200-site blocks.
    """
    if reads.n_sites == 0:
        raise InputError(f"{reads.contig}: no biallelic SNPs to analyse")
    contig = reads.contig
    input_positions, dropped = (reads.positions, dict(reads.dropped))
    # with them the read sharing between nearby sites (4.21), from the depths of the sites the sample QC reads
    site, samples, sharing = filter_sites(
        reads.positions,
        reads.counts,
        config.site_quality,
        config.sample_quality,
        contig=contig,
        team=team,
        map_work=map_work,
        reverse=reverse,
    )
    if site.kept.all():
        kept = reads
    else:
        indices = np.flatnonzero(site.kept)
        kept = replace(reads, positions=reads.positions[indices], counts=np.take(reads.counts, indices, axis=1))
        del indices
    del reads
    if kept.n_sites == 0:
        raise InputError(f"{contig}: the site filter kept no sites")
    counts, positions = (kept.counts, kept.positions)
    fitted, report = calibration.calibrate(counts, positions, config.read_model, map_work=map_work)
    model = fitted if fitted is not None else fixed_model(config.read_model.fixed_error, kept.n_samples)
    error, heterozygote = model.error_and_heterozygote()
    likelihoods = build_likelihoods(
        counts,
        error,
        heterozygote,
        model.heterozygote_inverse_concentration,
        config.read_model.lookup_depth,
        map_work=map_work,
    )
    total_depth = counts.sum(axis=(0, 2), dtype=np.int64)
    discovery_sites = total_depth >= config.discovery_sites.threshold()
    founder_excluded = np.zeros(kept.n_samples, bool) if samples is None else np.asarray(samples.flagged, bool)
    evidence_counts: np.ndarray | PairArray
    observed: np.ndarray | PairArray
    if isinstance(likelihoods, CodedLikelihoods):
        evidence_counts, observed = pair_views(likelihoods, counts)
    else:
        evidence_counts, observed = counts, _observation_mask(counts)
    # the cohort of the founder view: QC-flagged libraries do not shape the outlier law
    cohort = cohort_genotype_frequencies(
        likelihoods, observed, kept.n_sites, excluded=founder_excluded, map_work=map_work
    )
    evidence = ContigEvidence(
        payload_identity="",
        cohort_frequencies=cohort,
        contig=contig,
        sample_ids=kept.sample_ids,
        positions=positions,
        counts=evidence_counts,
        likelihoods=likelihoods,
        observed=observed,
        discovery_sites=discovery_sites,
        founder_excluded=founder_excluded,
        block_spans=_block_spans(kept.n_sites, config.block_sites),
    )
    return PreparedEvidence(
        evidence,
        EvidenceDecisions(
            contig=contig,
            input_positions=input_positions,
            dropped=dropped,
            site_quality=site,
            sample_quality=samples,
            read_model=report,
        ),
        sharing,
    )


__all__ = ["PreparedEvidence", "prepare_contig"]
