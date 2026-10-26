"""Observations of one contig: allele depths in, genotype likelihoods out.

Two types cross the stage boundary:

* :class:`ContigReads` -- the input of the evidence stage: REF/ALT read depths of the analysed
  samples at the contig's biallelic SNPs. Produced by the VCF loader
  (:func:`haplotype_reconstruction.evidence.vcf.read_contig`) or by the simulation source
  (:class:`haplotype_reconstruction.simulation.source.SimulationSource`).
* :class:`ContigEvidence` -- the product of the evidence stage (record ``evidence``): the site
  axis after site-quality filtering, the counts, the normalised raw genotype likelihoods of the
  fitted read model, the observation mask, the discovery sites, the per-contig contamination
  flags and the discovery block partition. Every later stage reads its evidence from this one
  record; nobody re-normalises or re-validates the likelihoods (they are validated once, when
  the evidence stage makes them).

Conventions: positions are int64 VCF POS (1-based), strictly increasing; the sample axis is the
analysed-sample order of the input (VCF header order, or the simulation's observed individuals in
pedigree order); genotype index g = number of ALT alleles (0, 1, 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np


# Reasons a loaded record is not part of ContigReads (keys of ContigReads.dropped).
DROPPED_NOT_SNP = "not_snp"                      # REF or ALT is not a single base (indel, MNP, symbolic, '*')
DROPPED_MULTIALLELIC = "multiallelic"            # more than one ALT allele
DROPPED_DUPLICATE_POSITION = "duplicate_position"  # >= 2 biallelic SNP records at one position: all dropped
DROPPED_REASONS = (DROPPED_NOT_SNP, DROPPED_MULTIALLELIC, DROPPED_DUPLICATE_POSITION)


def _frozen(array, dtype):
    value = np.ascontiguousarray(array, dtype=dtype)
    if value is array and value.flags.writeable:
        value = value.view()
    value.flags.writeable = False
    return value


@dataclass(frozen=True)
class ContigReads:
    """Allele depths of one contig's biallelic SNPs (the input of the evidence stage).

    contig: contig name.
    sample_ids: analysed samples, in the order of the counts' first axis.
    positions: int64[L] VCF POS (1-based), strictly increasing (one record per position).
    counts: int32[N, L, 2] read depths supporting (REF, ALT); a missing AD value is 0 reads.
    dropped: records not in this object, by reason (``DROPPED_REASONS``): loader policy for VCF
        input (records that are not biallelic SNPs, and every record of a position that has two or
        more biallelic SNP records); empty for simulated input.
    """

    contig: str
    sample_ids: tuple[str, ...]
    positions: np.ndarray
    counts: np.ndarray
    dropped: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "sample_ids", tuple(map(str, self.sample_ids)))
        object.__setattr__(self, "positions", _frozen(self.positions, np.int64))
        object.__setattr__(self, "counts", _frozen(self.counts, np.int32))
        object.__setattr__(self, "dropped", MappingProxyType(
            {str(reason): int(count) for reason, count in dict(self.dropped).items()}))

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def n_sites(self) -> int:
        return len(self.positions)

    def select_sites(self, keep: np.ndarray) -> "ContigReads":
        """The same reads at the sites where ``keep`` (bool[L]) is True."""
        keep = np.asarray(keep, bool)
        return replace(self, positions=self.positions[keep], counts=self.counts[:, keep])


@dataclass(frozen=True)
class ContigEvidence:
    """One contig's evidence (record ``evidence``): the stage boundary every later stage reads.

    contig, sample_ids: as in :class:`ContigReads` (N samples).
    positions: int64[L] sites kept by the site-quality filter (all input sites when the filter is
        off or not applied), strictly increasing.
    counts: int32[N, L, 2] (REF, ALT) read depths at these sites, or None when the record was
        loaded without them (``evidence.records.load(..., counts=False)``; most stages need only
        likelihoods and the observation mask).
    likelihoods: float64[N, L, 3] P(reads | genotype g) under the contig's fitted read model (the
        fixed binomial model when calibration falls back), normalised over g for every cell; a
        zero-depth cell is exactly (1/3, 1/3, 1/3). No genotype prior is applied.
    observed: bool[N, L] positive read depth (REF + ALT > 0).
    discovery_sites: bool[L] sites the discovery search uses: total depth over all N samples at
        least ``DiscoverySiteRule.minimum_total_depth`` (5) reads, whatever the cohort size
        (``evidence.config.DiscoverySiteRule``; base loader ``keep_flags``).
    founder_excluded: bool[N] samples flagged by the contamination QC on this contig (all False
        when the QC is off or did not apply). Founder construction sees them unobserved
        (:meth:`founder_view`); painting and later stages use every sample's evidence.
    block_spans: int64[B, 2] discovery blocks as [start, stop) indices into the site axis:
        consecutive non-overlapping windows of ``EvidenceConfig.block_sites`` sites (200; the last
        block may be shorter).
    """

    contig: str
    sample_ids: tuple[str, ...]
    positions: np.ndarray
    counts: np.ndarray | None
    likelihoods: np.ndarray
    observed: np.ndarray
    discovery_sites: np.ndarray
    founder_excluded: np.ndarray
    block_spans: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "sample_ids", tuple(map(str, self.sample_ids)))
        object.__setattr__(self, "positions", _frozen(self.positions, np.int64))
        if self.counts is not None:
            object.__setattr__(self, "counts", _frozen(self.counts, np.int32))
        object.__setattr__(self, "likelihoods", _frozen(self.likelihoods, np.float64))
        object.__setattr__(self, "observed", _frozen(self.observed, np.bool_))
        object.__setattr__(self, "discovery_sites", _frozen(self.discovery_sites, np.bool_))
        object.__setattr__(self, "founder_excluded", _frozen(self.founder_excluded, np.bool_))
        object.__setattr__(self, "block_spans", _frozen(
            np.asarray(self.block_spans, np.int64).reshape(-1, 2), np.int64))

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def n_sites(self) -> int:
        return len(self.positions)

    @property
    def n_blocks(self) -> int:
        return len(self.block_spans)

    def block(self, index: int) -> slice:
        """Site-axis slice of discovery block ``index``."""
        start, stop = self.block_spans[index]
        return slice(int(start), int(stop))

    def founder_view(self) -> "ContigEvidence":
        """Founder construction's missing-data view (discovery through founder refit).

        Excluded rows have no observations. Immutable counts and likelihoods are shared:
        consumers must mask unobserved cells before reading them, or materialise their
        neutral values (zero counts, likelihoods 1/3) in the block/view they already copy.
        Site axis, discovery sites and blocks are unchanged. No exclusions returns self.
        """
        if not self.founder_excluded.any():
            return self
        from ..execution import reserve_memory
        with reserve_memory(self.observed.nbytes, label="founder observation mask"):
            observed = np.array(self.observed, copy=True)
            observed[self.founder_excluded] = False
        return replace(self, observed=observed)
