"""Local founder panels: the product of local selection that assembly reads.

One :class:`LocalPanel` per informative discovery block (a block with at least one discovery
site observed in at least one sample), in genomic order. A panel holds the K founder rows that
the block's local path selection kept, their released alleles and the per-sample unknown-copy
summary that gates assembly's completion fills. Nothing else: search traces, fitted
frequencies, likelihood memos and release statistics live in local selection's own records.

Conventions: positions int64 VCF POS (1-based); alleles int8 0 = REF, 1 = ALT, -1 = not
released; the sample axis is the evidence sample order (``ContigEvidence.sample_ids``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalPanel:
    """The selected founder panel of one discovery block (exactly what assembly reads).

    block: index of the block in the contig's discovery blocks (``ContigEvidence.block_spans``).
    site_start: contig site index of the block's first site; the panel covers the contig sites
        ``site_start .. site_start + L - 1`` (``block_spans[block]``).
    positions: int64[L] positions of all sites of the block.
    kept: bool[L] discovery sites (``ContigEvidence.discovery_sites`` on the block); the panel was
        fitted on these sites only.
    calls: int8[K, L] released founder alleles; -1 where the release rule withheld the allele and
        at every site that is not a discovery site.
    alt_probability: float64[K, L] fitted probability that the founder carries ALT (the one-bit
        conditional posterior of the fitted path model); 0.5 at sites that are not discovery sites.
    support: float64[K, L] favourable carriers behind each release decision (the number of
        samples whose own evidence favours the allele and that more likely than not carry the
        founder there); 0 at sites that are not discovery sites.
    has_depth: bool[N] the sample has an observed discovery site in the block.
    unknown_slots: int8[N] per sample, the largest marginal MAP count (0, 1, 2) of copies drawn
        from the unknown state over its observed discovery sites; 2 for a sample without depth.
    unknown_mass: float, ``unknown_slots[has_depth].sum() / max(2 * has_depth.sum(), 1)``: the
        spatial MAP unknown-copy fraction that gates assembly's completion fills (base
        ``wildcard_mass``); not a probability of any unknown state.
    """

    block: int
    site_start: int
    positions: np.ndarray
    kept: np.ndarray
    calls: np.ndarray
    alt_probability: np.ndarray
    support: np.ndarray
    has_depth: np.ndarray
    unknown_slots: np.ndarray
    unknown_mass: float

    @property
    def founders(self) -> int:
        """K, the number of founder rows."""
        return int(self.calls.shape[0])

    @property
    def sites(self) -> int:
        """L, the number of sites of the block."""
        return int(self.positions.shape[0])


def unknown_copy_mass(unknown_slots: np.ndarray, has_depth: np.ndarray) -> float:
    """The spatial MAP unknown-copy fraction of a block: unknown copies of the samples with
    depth over their 2 copies each (0 when no sample has depth)."""
    has_depth = np.asarray(has_depth, dtype=bool)
    return float(np.asarray(unknown_slots)[has_depth].sum() / max(2 * int(has_depth.sum()), 1))
