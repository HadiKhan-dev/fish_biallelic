"""Pairwise founder sharing: a pure view of released paintings (PR20), and its product view.

Per pair of samples and painting bin, the founder rows the two samples' released tracks share: 0, 1 or 2, the
multiset intersection of their two named rows, counted only where all four tracks are SINGLETON_NAMED. Every other
bin of the pair is classified by why it is not counted. Lengths are the bins' Morgans on the working map. Rows are
component-local, and both samples of a pair share the component's bins and panel, so no row identity across
components or label-continuity links is needed. Pure NumPy views: no I/O, storage or scheduling; nothing here feeds
back into any stage.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from ..core.errors import InvariantError
from .genetic_map import ChromosomeGeneticMap
from .painting import ComponentPainting, Painting, TrackStatus
from .roles import freeze_record_arrays, record_field

__all__ = [
    "NAMED",
    "CANDIDATES",
    "UNIDENTIFIED",
    "ABSTAINED",
    "TRACK_CATEGORY",
    "LENGTHS",
    "PairSharing",
    "bin_morgans",
    "component_pair_lengths",
    "contig_pair_lengths",
    "pair_index",
]

# Why a track does or does not name one founder row; a pair-bin takes the largest code of its four tracks.
NAMED = 0  # SINGLETON_NAMED: one founder row
CANDIDATES = 1  # pooled or local allele equivalence, or a switch inside the bin: a set of candidate rows
UNIDENTIFIED = 2  # background (outside the panel) or a class with no called allele
ABSTAINED = 3  # no evidence or low posterior: the painting makes no call

TRACK_CATEGORY = np.zeros(len(TrackStatus), dtype=np.uint8)
TRACK_CATEGORY[TrackStatus.INELIGIBLE_NO_EVIDENCE] = ABSTAINED
TRACK_CATEGORY[TrackStatus.LOW_POSTERIOR_ABSTENTION] = ABSTAINED
TRACK_CATEGORY[TrackStatus.SINGLETON_NAMED] = NAMED
TRACK_CATEGORY[TrackStatus.POOLED_EQUIVALENCE] = CANDIDATES
TRACK_CATEGORY[TrackStatus.BACKGROUND] = UNIDENTIFIED
TRACK_CATEGORY[TrackStatus.UNANCHORED_TRAJECTORY] = UNIDENTIFIED
TRACK_CATEGORY[TrackStatus.LOCAL_ALLELE_EQUIVALENCE] = CANDIDATES
TRACK_CATEGORY[TrackStatus.SWITCH_INTERVAL] = CANDIDATES
TRACK_CATEGORY.flags.writeable = False

# Columns of the per-pair length arrays (Morgans): jointly named length sharing 0, 1 and 2 founder rows, then the
# painted length not counted, by the pair-bin's category.
LENGTHS = ("shared0", "shared1", "shared2", "candidates", "unidentified", "abstained")


def pair_index(n_samples: int) -> NDArray[np.int64]:
    """int64[P, 2] the pairs (i, j), i < j, in the row-major order of every per-pair array; P = N(N-1)/2."""
    first, second = np.triu_indices(n_samples, 1)
    return np.stack((first, second), axis=1).astype(np.int64)


def bin_morgans(edges: NDArray[np.int64], genetic_map: ChromosomeGeneticMap) -> NDArray[np.float64]:
    """float64[B] the working-map length of each bin [edges[b], edges[b + 1]) (Morgans; fallback rate off-map)."""
    return np.diff(np.asarray(genetic_map.cumulative_morgans(np.asarray(edges, dtype=np.float64)), dtype=np.float64))


def component_pair_lengths(
    component: ComponentPainting, weights: NDArray[np.float64], out: NDArray[np.float64]
) -> None:
    """Add one component's per-pair lengths (``weights`` float64[B] per bin) to ``out`` float64[P, 6].

    Row passes over the pairs (i, j > i): O(N^2 B) time, O(N B) workspace; the pair-bin code is the shared-row count
    (0, 1, 2) when all four tracks are named, else 2 + the pair's category, and ``np.bincount`` sums the weights per
    pair and code.
    """
    n_samples, _, n_bins = component.labels.shape
    if n_bins == 0 or n_samples < 2:
        return
    if weights.shape != (n_bins,) or out.shape != (n_samples * (n_samples - 1) // 2, len(LENGTHS)):
        raise InvariantError("pair-sharing weights or output do not match the component")
    category = np.maximum(TRACK_CATEGORY[component.status[:, 0]], TRACK_CATEGORY[component.status[:, 1]])
    low = np.minimum(component.labels[:, 0], component.labels[:, 1])
    high = np.maximum(component.labels[:, 0], component.labels[:, 1])
    tiled = np.tile(np.asarray(weights, dtype=np.float64), n_samples - 1)  # first n * B entries serve n rows
    columns = len(LENGTHS)
    start = 0
    for i in range(n_samples - 1):
        n = n_samples - 1 - i
        pair_category = np.maximum(category[i + 1 :], category[i])
        # Multiset intersection of {low, high} pairs: the larger of the straight and crossed matches.
        straight = (low[i + 1 :] == low[i]).view(np.uint8) + (high[i + 1 :] == high[i]).view(np.uint8)
        crossed = (high[i + 1 :] == low[i]).view(np.uint8) + (low[i + 1 :] == high[i]).view(np.uint8)
        np.maximum(straight, crossed, out=straight)
        code = np.where(pair_category == NAMED, straight, pair_category + np.uint8(2)).astype(np.intp)
        code += (columns * np.arange(n, dtype=np.intp))[:, None]
        out[start : start + n] += np.bincount(code.ravel(), weights=tiled[: n * n_bins], minlength=columns * n).reshape(
            n, columns
        )
        start += n


def contig_pair_lengths(painting: Painting, genetic_map: ChromosomeGeneticMap) -> NDArray[np.float64]:
    """float64[P, 6] per-pair lengths (Morgans, columns LENGTHS) of one painted contig, summed over its components."""
    n_samples = len(painting.sample_ids)
    out = np.zeros((n_samples * (n_samples - 1) // 2, len(LENGTHS)), dtype=np.float64)
    for component in painting.components:
        component_pair_lengths(component, bin_morgans(component.bins.edges, genetic_map), out)
    return out


@dataclass(frozen=True)
class PairSharing:
    """Founder sharing of every sample pair over ``contigs`` (a view computed from released paintings; never stored).

    lengths: float64[P, 6] Morgans per pair (i, j), i < j in ``sample_ids`` order (:func:`pair_index`), columns
        LENGTHS: jointly named length on which the pair shares 0, 1 and 2 founder rows; painted length not counted
        because a track names candidate rows, is unidentified, or abstains (the pair's worst track).
    """

    __payload_view__: ClassVar[bool] = True
    sample_ids: tuple[str, ...] = record_field("obs", axes=("N",), dtype="str", stored=False)
    contigs: tuple[str, ...] = record_field("obs", axes=("C",), dtype="str", stored=False)
    lengths: NDArray[np.float64] = record_field(
        "report", axes=("P", "6"), units="Morgans", dtype="float64", stored=False
    )

    def __post_init__(self) -> None:
        n_samples = len(self.sample_ids)
        if self.lengths.shape != (n_samples * (n_samples - 1) // 2, len(LENGTHS)):
            raise InvariantError("pair-sharing lengths do not match the sample axis")
        freeze_record_arrays(self)

    @classmethod
    def total(cls, parts: Iterable[PairSharing]) -> PairSharing | None:
        """The sum over parts (in their order) with one sample axis and distinct contigs; None for no parts.

        Parts are consumed one at a time, so a generator keeps one contig's arrays alive besides the running sum.
        """
        sample_ids: tuple[str, ...] | None = None
        contigs: list[str] = []
        lengths = None
        for part in parts:
            if sample_ids is None:
                sample_ids, lengths = part.sample_ids, np.array(part.lengths, dtype=np.float64)
            elif part.sample_ids != sample_ids:
                raise InvariantError("pair-sharing parts have different sample axes")
            else:
                lengths += part.lengths
            contigs.extend(part.contigs)
        if sample_ids is None or lengths is None:
            return None
        if len(set(contigs)) != len(contigs):
            raise InvariantError("pair-sharing parts repeat a contig")
        return cls(sample_ids, tuple(contigs), lengths)

    def pairs(self) -> NDArray[np.int64]:
        """int64[P, 2] sample indices (i, j), i < j, of each row."""
        return pair_index(len(self.sample_ids))

    def named_morgans(self) -> NDArray[np.float64]:
        """float64[P] jointly named length (Morgans): the denominator of the shares."""
        return self.lengths[:, :3].sum(axis=1)

    def painted_morgans(self) -> NDArray[np.float64]:
        """float64[P] painted length (Morgans); the same for every pair of one painting."""
        return self.lengths.sum(axis=1)

    def shares(self) -> NDArray[np.float64]:
        """float64[P, 3] IBD0, IBD1 and IBD2 shares of the jointly named length; NaN where it is zero."""
        named = self.named_morgans()
        out = np.full((len(named), 3), np.nan)
        np.divide(self.lengths[:, :3], named[:, None], out=out, where=named[:, None] > 0)
        return out

    def coverage(self) -> NDArray[np.float64]:
        """float64[P, 4] shares of the painted length: jointly named, candidates, unidentified, abstained (sum 1;
        NaN where nothing is painted)."""
        painted = self.painted_morgans()
        out = np.full((len(painted), 4), np.nan)
        counted = np.column_stack((self.named_morgans(), self.lengths[:, 3:]))
        np.divide(counted, painted[:, None], out=out, where=painted[:, None] > 0)
        return out
