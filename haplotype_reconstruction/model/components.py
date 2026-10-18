"""Assembly products: founder components as row paths over immutable prepared panels.

Assembly never copies founder alleles into new chromosome arrays. Its atomic unit is the
prepared panel of one block (the path-selected local panel after founder completion); every
assembly product (a joined batch, a hierarchy level, a refined component) is a row-path matrix
``rows[f, b]`` naming, for founder ``f``, the local row of prepared panel ``b`` that the founder
copies there. Chromosome-scale arrays are derived on demand by :func:`materialise`.

Conventions (all components of one assembly):
* prepared panels are in genomic order; each is a contiguous window of the contig's site axis;
* components partition the prepared panels ``0 .. len(prepared) - 1`` into consecutive ranges
  ``[start, stop)`` in genomic order;
* founder labels are row indices ``0 .. K - 1`` of a component (no separate key objects);
* alleles are int8: 0 = REF, 1 = ALT, -1 = no call; positions are int64 VCF POS (1-based).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
import hashlib
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PreparedPanel:
    """One block's local founder panel after founder completion (immutable; R local rows).

    block: index of the block in the contig's evidence block spans (``ContigEvidence.blocks``).
    site_start: contig site index of the block's first site; the panel covers the contig sites
        ``site_start .. site_start + L - 1``.
    positions: int64[L] genomic positions, strictly increasing.
    kept: bool[L] discovery sites (base ``keep_flags``); only kept sites enter a likelihood.
    released: int8[R, L] published founder alleles: the local-selection calls plus the completion
        fills (fills only at kept sites; calls at not-kept sites are carried as local selection
        released them).
    inference: int8[R, L] the frozen pre-fill snapshot: local-selection calls with not-kept sites
        set to -1. Every likelihood of assembly, founder refinement and painting reads these, and so
        does the count rule's row geometry (nearest rows, shadows, novelty), so a completion fill
        never feeds back into its own carrier evidence.
    alt_probability: float64[R, L] ALT pseudo-probability of each founder allele (local
        selection's value; the completion posterior at a fill; 0.5 where uncalled).
    support: float64[R, L] directional site supporters (local selection; carried for reports).
    """

    block: int
    site_start: int
    positions: np.ndarray
    kept: np.ndarray
    released: np.ndarray
    inference: np.ndarray
    alt_probability: np.ndarray
    support: np.ndarray

    @property
    def rows(self) -> int:
        """R, the number of local founder rows."""
        return int(self.released.shape[0])

    @property
    def sites(self) -> int:
        """L, the number of sites of the block."""
        return int(self.positions.shape[0])

    @property
    def filled(self) -> np.ndarray:
        """bool[R, L] completion fills: published calls without a pre-fill call at kept sites."""
        return (self.released >= 0) & (self.inference < 0) & self.kept[None, :]

    @cached_property
    def digest(self) -> str:
        """SHA-256 of everything a join or refinement reads from the panel (site range, positions,
        kept mask, released and inference calls): the content identity used for join reuse."""
        digest = hashlib.sha256(f"prepared-panel:{int(self.site_start)}".encode())
        for array in (self.positions, self.kept, self.released, self.inference):
            value = np.ascontiguousarray(array)
            digest.update(f"{value.dtype.str}{value.shape}".encode())
            digest.update(value.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class CountStepDecision:
    """One decision of a component's final count step (founder refinement at the last level).

    kind: "deletion" or "addition".
    founders: panel size K when the round ran (the incumbent's founder count).
    accept_larger: True keeps the larger panel (deletion rejected / addition accepted).
    skipped_by_bound: the round was decided by an optimistic bound without refits.
    margin: min(m_d, m_q) of the effective-carrier test in units of the per-founder cost
        (None when skipped by a bound).
    band: bootstrap percentile interval of that margin (None when skipped by a bound).
    ambiguous: the band contains zero in its interior.
    """

    kind: str
    founders: int
    accept_larger: bool
    skipped_by_bound: bool
    margin: float | None
    band: tuple[float, float] | None
    ambiguous: bool


@dataclass(frozen=True)
class CountCall:
    """How firmly one component's founder count is determined (its last count step).

    founders / founders_before: K after / before the count step (None when no step ran).
    ambiguous: some decision's band contains zero, the step cycled, or the restored best-F panel
        fails its own deletion test (founder refinement's definition).
    count_cycle: an addition was followed by a deletion of a row the same search created.
    cycle_undecided: that cycle began with a decided deletion and closed ambiguously.
    decisions: the step's decisions in order.
    reason: why no count step ran ("single_block_or_founder"), else None.
    """

    founders: int | None
    founders_before: int | None
    ambiguous: bool
    count_cycle: bool
    cycle_undecided: bool
    decisions: tuple[CountStepDecision, ...]
    reason: str | None = None


@dataclass(frozen=True)
class FounderComponent:
    """One phase component: K founder paths through the prepared panels ``start .. stop - 1``.

    start, stop: prepared-panel range [start, stop) (indices into the assembly's prepared panels).
    rows: int32[K, B], B = stop - start; ``rows[f, b]`` is the local row of prepared panel
        ``start + b`` that founder f copies. Founder labels are the row indices 0 .. K - 1.
    break_before, break_after: reason of a data-unsupported phase break at the component's left /
        right boundary ("insufficient_joint_informative_samples": fewer than the configured number
        of samples are informative on both sides); None at a chromosome end and at a boundary the
        hierarchy left unjoined for another reason (a batch edge at the last level, a batch without
        informative samples).
    count_call: the last count step's call (final assembly, after founder refinement), else None.
    """

    start: int
    stop: int
    rows: np.ndarray
    break_before: str | None = None
    break_after: str | None = None
    count_call: CountCall | None = None

    @property
    def founders(self) -> int:
        return int(self.rows.shape[0])

    @property
    def blocks(self) -> int:
        return int(self.stop - self.start)


@dataclass(frozen=True)
class ComponentPanel:
    """A component's chromosome-scale founder arrays (built by :func:`materialise`).

    index: component index in genomic order.
    start, stop, rows, break_before, break_after, count_call: as in :class:`FounderComponent`.
    positions: int64[L] positions of the component's sites.
    released: int8[K, L] published alleles (with completion fills; founder refit replaces them).
    inference: int8[K, L] the frozen pre-fill snapshot (what painting and pedigree read).
    kept: bool[L] discovery sites.
    alt_probability: float64[K, L]; support: float64[K, L].
    atomic_offsets: int64[B] site offset of each prepared panel (atomic span) inside the
        component (atomic_offsets[0] = 0; span b ends where span b + 1 starts, the last at L).
    atomic_site_starts: int64[B] contig site index of each span's first site (a span is a
        contiguous window of the contig's sites; consecutive spans need not be adjacent when a
        block between them has no panel).
    """

    index: int
    start: int
    stop: int
    positions: np.ndarray
    released: np.ndarray
    inference: np.ndarray
    kept: np.ndarray
    alt_probability: np.ndarray
    support: np.ndarray
    atomic_offsets: np.ndarray
    atomic_site_starts: np.ndarray
    rows: np.ndarray
    break_before: str | None = None
    break_after: str | None = None
    count_call: CountCall | None = None

    @property
    def founders(self) -> int:
        return int(self.released.shape[0])

    @property
    def site_start(self) -> int:
        """Contig site index of the component's first site."""
        return int(self.atomic_site_starts[0])

    def site_indices(self) -> np.ndarray:
        """int64[L] contig site index of every site of the component."""
        counts = self.atomic_counts
        return (np.repeat(self.atomic_site_starts - self.atomic_offsets, counts)
                + np.arange(len(self.positions), dtype=np.int64))

    @property
    def atomic_counts(self) -> np.ndarray:
        """int64[B] sites of each atomic span."""
        return np.diff(np.append(self.atomic_offsets, len(self.positions)))

    @property
    def source_rows(self) -> np.ndarray:
        """int32[K, L]: the prepared-panel row each founder copies at each site. Two founders with
        the same source row at a site share one latent allele there."""
        return np.repeat(self.rows, self.atomic_counts, axis=1)


ComponentPanels = tuple[ComponentPanel, ...]
PreparedPanels = tuple[PreparedPanel, ...]
FounderComponents = tuple[FounderComponent, ...]

# Founder-axis arrays of a ComponentPanel, by axis layout: per site (K, L) or per span (K, B).
_SITE_FOUNDER_FIELDS = ("released", "inference", "alt_probability", "support")
_SPAN_FOUNDER_FIELDS = ("rows",)


def materialise_one(index: int, component: FounderComponent,
                    prepared: Sequence[PreparedPanel]) -> ComponentPanel:
    """The chromosome-scale arrays of one component (see :func:`materialise`)."""
    panels = prepared[component.start:component.stop]
    rows = np.asarray(component.rows, dtype=np.int32)
    lengths = np.fromiter((panel.sites for panel in panels), dtype=np.int64, count=len(panels))
    offsets = np.zeros(len(panels), dtype=np.int64)
    np.cumsum(lengths[:-1], out=offsets[1:])

    def gather(name):
        return np.concatenate([getattr(panel, name)[rows[:, b]]
                               for b, panel in enumerate(panels)], axis=1)

    return ComponentPanel(
        index=int(index), start=int(component.start), stop=int(component.stop),
        positions=np.concatenate([panel.positions for panel in panels]),
        released=gather("released"), inference=gather("inference"),
        kept=np.concatenate([panel.kept for panel in panels]),
        alt_probability=gather("alt_probability"), support=gather("support"),
        atomic_offsets=offsets,
        atomic_site_starts=np.fromiter((panel.site_start for panel in panels), dtype=np.int64,
                                       count=len(panels)),
        rows=rows, break_before=component.break_before,
        break_after=component.break_after, count_call=component.count_call)


def materialise(components: Sequence[FounderComponent],
                prepared: Sequence[PreparedPanel]) -> ComponentPanels:
    """Chromosome-scale founder arrays of every component, in order.

    A founder's alleles at the sites of prepared panel b are that panel's row ``rows[f, b]``
    (released and inference calls, ALT probability, supporters).
    """
    return tuple(materialise_one(index, component, prepared)
                 for index, component in enumerate(components))


def relabel_rows(panel: ComponentPanel, row_maps) -> ComponentPanel:
    """Relabel founders span by span: in atomic span a, new founder k carries old founder
    ``row_maps[a, k]`` (row_maps: int[B, K], each row a permutation of 0 .. K - 1).

    Every founder-axis array is permuted (per-site arrays within each span's sites, the row
    paths per span); positions, kept, offsets, breaks and the count call are unchanged.
    """
    row_maps = np.asarray(row_maps, dtype=np.int64)
    counts = panel.atomic_counts
    site_map = np.repeat(row_maps.T, counts, axis=1)          # (K, L): old founder per new k
    site_index = np.arange(len(panel.positions))[None, :]
    changes = {name: getattr(panel, name)[site_map, site_index] for name in _SITE_FOUNDER_FIELDS}
    span_index = np.arange(len(counts))[None, :]
    for name in _SPAN_FOUNDER_FIELDS:
        changes[name] = getattr(panel, name)[row_maps.T, span_index]
    return replace(panel, **changes)
