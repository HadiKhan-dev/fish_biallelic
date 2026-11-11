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

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .roles import record_field

__all__ = [
    "PreparedPanel",
    "CountStepDecision",
    "CountCall",
    "FounderComponent",
    "ComponentPanel",
    "LocalPanel",
    "PartnerChange",
    "ComponentPanels",
    "PreparedPanels",
    "FounderComponents",
    "materialise_one",
    "materialise",
    "relabel_rows",
    "partner_changes",
    "unknown_copy_mass",
    "SITE_FOUNDER_FIELDS",
    "SPAN_FOUNDER_FIELDS",
    "REASON",
    "METHOD",
]


def _readonly(array: np.ndarray) -> np.ndarray:
    value = array.view()
    value.flags.writeable = False
    return value


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

    block: int = record_field("state", axes=(), units="block", dtype="int")
    site_start: int = record_field("state", axes=(), units="site", dtype="int")
    positions: NDArray[np.int64] = record_field("obs", axes=("L",), units="bp", dtype="int64")
    kept: NDArray[np.bool_] = record_field("state", axes=("L",), units="1", dtype="bool")
    released: NDArray[np.int8] = record_field("call", axes=("R", "L"), units="1", dtype="int8")
    inference: NDArray[np.int8] = record_field("state", axes=("R", "L"), units="1", dtype="int8")
    alt_probability: NDArray[np.float64] = record_field("state", axes=("R", "L"), units="probability", dtype="float64")
    support: NDArray[np.float64] = record_field("report", axes=("R", "L"), units="count", dtype="float64")

    def __post_init__(self) -> None:
        for name in ("positions", "kept", "released", "inference", "alt_probability", "support"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

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

    kind: str = record_field("call", axes=(), units="1", dtype="str")
    founders: int = record_field("call", axes=(), units="1", dtype="int")
    accept_larger: bool = record_field("call", axes=(), units="1", dtype="bool")
    skipped_by_bound: bool = record_field("call", axes=(), units="1", dtype="bool")
    margin: float | None = record_field("call", axes=(), units="1", dtype="float | None")
    band: tuple[float, float] | None = record_field("call", axes=(), units="1", dtype="tuple[float, float] | None")
    ambiguous: bool = record_field("call", axes=(), units="1", dtype="bool")


@dataclass(frozen=True)
class CountCall:
    """How firmly one component's founder count is determined (its last count step).

    founders / founders_before: K after / before the count step (None when no step ran).
    ambiguous: some decision's band contains zero, or the restored best-F panel fails its own
        deletion test (founder refinement's definition).
    decisions: the step's decisions in order.
    reason: why no count step ran ("single_block_or_founder"), else None.
    """

    founders: int | None = record_field("call", axes=(), units="1", dtype="int | None")
    founders_before: int | None = record_field("call", axes=(), units="1", dtype="int | None")
    ambiguous: bool = record_field("call", axes=(), units="1", dtype="bool")
    decisions: tuple[CountStepDecision, ...] = record_field(
        "call", axes=(), units="1", dtype="tuple[CountStepDecision, ...]"
    )
    reason: str | None = record_field("call", axes=(), units="1", dtype="str | None", default=None)


@dataclass(frozen=True)
class FounderComponent:
    """One phase component: K founder paths through the prepared panels ``start .. stop - 1``.

    start, stop: prepared-panel range [start, stop) (indices into the assembly's prepared panels).
    rows: int32[K, B], B = stop - start; ``rows[f, b]`` is the local row of prepared panel
        ``start + b`` that founder f copies. Founder labels are the row indices 0 .. K - 1.
    break_before, break_after: reason of a phase break at the component's left /
        right boundary ("insufficient_joint_informative_samples": fewer than the configured number
        of samples are informative on both sides; "unresolved_beam_materialization": no full joined
    incumbent was established within the complete-family beam bound); None at a chromosome end and at a boundary the
        hierarchy left unjoined for another reason (a batch edge at the last level, a batch without
        informative samples).
    count_call: the last count step's call (final assembly, after founder refinement), else None.
    """

    start: int = record_field("call", axes=(), units="block", dtype="int")
    stop: int = record_field("call", axes=(), units="block", dtype="int")
    rows: NDArray[np.int32] = record_field("call", axes=("K", "B"), units="1", dtype="int32")
    break_before: str | None = record_field("call", axes=(), units="1", dtype="str | None", default=None)
    break_after: str | None = record_field("call", axes=(), units="1", dtype="str | None", default=None)
    count_call: CountCall | None = record_field("call", axes=(), units="1", dtype="CountCall | None", default=None)

    def __post_init__(self) -> None:
        for name in ("rows",):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

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

    __payload_view__: ClassVar[bool] = True

    index: int = record_field("state", axes=(), units="1", dtype="int", stored=False)
    start: int = record_field("call", axes=(), units="block", dtype="int", stored=False)
    stop: int = record_field("call", axes=(), units="block", dtype="int", stored=False)
    positions: NDArray[np.int64] = record_field("obs", axes=("L",), units="bp", dtype="int64", stored=False)
    released: NDArray[np.int8] = record_field("call", axes=("K", "L"), units="1", dtype="int8", stored=False)
    inference: NDArray[np.int8] = record_field("state", axes=("K", "L"), units="1", dtype="int8", stored=False)
    kept: NDArray[np.bool_] = record_field("state", axes=("L",), units="1", dtype="bool", stored=False)
    alt_probability: NDArray[np.float64] = record_field(
        "state", axes=("K", "L"), units="probability", dtype="float64", stored=False
    )
    support: NDArray[np.float64] = record_field("report", axes=("K", "L"), units="count", dtype="float64", stored=False)
    atomic_offsets: NDArray[np.int64] = record_field("state", axes=("B",), units="site", dtype="int64", stored=False)
    atomic_site_starts: NDArray[np.int64] = record_field(
        "state", axes=("B",), units="site", dtype="int64", stored=False
    )
    rows: NDArray[np.int32] = record_field("call", axes=("K", "B"), units="1", dtype="int32", stored=False)
    break_before: str | None = record_field("call", axes=(), units="1", dtype="str | None", stored=False, default=None)
    break_after: str | None = record_field("call", axes=(), units="1", dtype="str | None", stored=False, default=None)
    count_call: CountCall | None = record_field(
        "call", axes=(), units="1", dtype="CountCall | None", stored=False, default=None
    )

    def __post_init__(self) -> None:
        for name in (
            "positions",
            "released",
            "inference",
            "kept",
            "alt_probability",
            "support",
            "atomic_offsets",
            "atomic_site_starts",
            "rows",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

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
        return np.repeat(self.atomic_site_starts - self.atomic_offsets, counts) + np.arange(
            len(self.positions), dtype=np.int64
        )

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
SITE_FOUNDER_FIELDS = ("released", "inference", "alt_probability", "support")
SPAN_FOUNDER_FIELDS = ("rows",)


def materialise_one(index: int, component: FounderComponent, prepared: Sequence[PreparedPanel]) -> ComponentPanel:
    """The chromosome-scale arrays of one component (see :func:`materialise`)."""
    panels = prepared[component.start : component.stop]
    rows = np.asarray(component.rows, dtype=np.int32)
    lengths = np.fromiter((panel.sites for panel in panels), dtype=np.int64, count=len(panels))
    offsets = np.zeros(len(panels), dtype=np.int64)
    np.cumsum(lengths[:-1], out=offsets[1:])

    def gather(name: str) -> np.ndarray:
        return np.concatenate([getattr(panel, name)[rows[:, b]] for b, panel in enumerate(panels)], axis=1)

    return ComponentPanel(
        index=int(index),
        start=int(component.start),
        stop=int(component.stop),
        positions=np.concatenate([panel.positions for panel in panels]),
        released=gather("released"),
        inference=gather("inference"),
        kept=np.concatenate([panel.kept for panel in panels]),
        alt_probability=gather("alt_probability"),
        support=gather("support"),
        atomic_offsets=offsets,
        atomic_site_starts=np.fromiter((panel.site_start for panel in panels), dtype=np.int64, count=len(panels)),
        rows=rows,
        break_before=component.break_before,
        break_after=component.break_after,
        count_call=component.count_call,
    )


def materialise(components: Sequence[FounderComponent], prepared: Sequence[PreparedPanel]) -> ComponentPanels:
    """Chromosome-scale founder arrays of every component, in order.

    A founder's alleles at the sites of prepared panel b are that panel's row ``rows[f, b]``
    (released and inference calls, ALT probability, supporters).
    """
    return tuple(materialise_one(index, component, prepared) for index, component in enumerate(components))


def relabel_rows(panel: ComponentPanel, row_maps: ArrayLike) -> ComponentPanel:
    """Relabel founders span by span: in atomic span a, new founder k carries old founder
    ``row_maps[a, k]`` (row_maps: int[B, K], each row a permutation of 0 .. K - 1).

    Every founder-axis array is permuted (per-site arrays within each span's sites, the row
    paths per span); positions, kept, offsets, breaks and the count call are unchanged.
    """
    row_maps = np.asarray(row_maps, dtype=np.int64)
    counts = panel.atomic_counts
    site_map = np.repeat(row_maps.T, counts, axis=1)  # (K, L): old founder per new k
    site_index = np.arange(len(panel.positions))[None, :]
    changes = {name: getattr(panel, name)[site_map, site_index] for name in SITE_FOUNDER_FIELDS}
    span_index = np.arange(len(counts))[None, :]
    for name in SPAN_FOUNDER_FIELDS:
        changes[name] = getattr(panel, name)[row_maps.T, span_index]
    return replace(panel, **changes)


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

    block: int = record_field("state", axes=(), units="block", dtype="int")
    site_start: int = record_field("state", axes=(), units="site", dtype="int")
    positions: NDArray[np.int64] = record_field("obs", axes=("L",), units="bp", dtype="int64")
    kept: NDArray[np.bool_] = record_field("state", axes=("L",), units="1", dtype="bool")
    calls: NDArray[np.int8] = record_field("call", axes=("K", "L"), units="1", dtype="int8")
    alt_probability: NDArray[np.float64] = record_field("state", axes=("K", "L"), units="probability", dtype="float64")
    support: NDArray[np.float64] = record_field("report", axes=("K", "L"), units="count", dtype="float64")
    has_depth: NDArray[np.bool_] = record_field("state", axes=("N",), units="1", dtype="bool")
    unknown_slots: NDArray[np.int8] = record_field("state", axes=("N",), units="count", dtype="int8")
    unknown_mass: float = record_field("state", axes=(), units="1", dtype="float")

    def __post_init__(self) -> None:
        for name in ("positions", "kept", "calls", "alt_probability", "support", "has_depth", "unknown_slots"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

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


REASON = "local_duplicate_partner_change"
METHOD = "duplicate-partner-v1"


@dataclass(frozen=True)
class PartnerChange:
    """Row ``row`` at the junction before block ``block`` (component-local, 1 .. B-1): its local rows on blocks
    block - 1 and block, and its partner founder rows there (disjoint, both non-empty)."""

    row: int = record_field("report", axes=(), units="1", dtype="int")
    block: int = record_field("report", axes=(), units="block", dtype="int")
    left_local_row: int = record_field("report", axes=(), units="1", dtype="int")
    right_local_row: int = record_field("report", axes=(), units="1", dtype="int")
    left_partners: tuple[int, ...] = record_field("report", axes=(), units="1", dtype="tuple[int, ...]")
    right_partners: tuple[int, ...] = record_field("report", axes=(), units="1", dtype="tuple[int, ...]")


def partner_changes(rows: ArrayLike) -> tuple[PartnerChange, ...]:
    """The duplicate-partner changes of a component's row paths ``rows`` int [K, B] (local row per founder and
    block), in (block, row) order. One pass over the blocks; O(B K^2) time, O(K^2) memory."""
    rows = np.asarray(rows)
    if rows.ndim != 2:
        raise ValueError("row paths must be a [founders, blocks] matrix")
    founders, blocks = rows.shape
    result = []
    previous = None
    for block in range(blocks):
        column = rows[:, block]
        partners = [tuple(int(j) for j in np.flatnonzero(column == column[k]) if j != k) for k in range(founders)]
        if previous is not None:
            for k in range(founders):
                left, right = previous[k], partners[k]
                if left and right and not set(left) & set(right):
                    result.append(PartnerChange(k, block, int(rows[k, block - 1]), int(column[k]), left, right))
        previous = partners
    return tuple(result)
