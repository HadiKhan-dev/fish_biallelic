"""Structural rearrangements of simulated genomes: polymorphic inversions, deletions and duplications, and a reference
whose order is wrong for every haplotype (a reversed, moved or unlinked segment).

Every coordinate is a reference bp coordinate (the axis of the VCF the pipeline reads); intervals are half-open
[start, end). Variant kinds:

* ``inversion``: carriers hold [start, end) in the orientation opposite to the reference;
* ``deletion``: carriers lack [start, end) (hemizygous or absent cells; no reads from that homologue);
* ``duplication``: carriers hold an extra copy of [start, end) at ``target`` (``target == end``: tandem). The copy is
  absent from the reference, so its reads align to [start, end): copy number raises the depth there, and the copy's
  paralogous differences appear as extra alleles;
* ``reversed`` / ``moved`` / ``unlinked`` (reference misorder, every haplotype): [start, end) truly lies reversed in
  place, between reference positions ``target - 1`` and ``target``, or on another chromosome.

Who carries a polymorphic variant is a property of stable source haplotypes (the cross founders' ancestors), looked up
at the variant's anchor (``start``; ``target`` for a duplication, the copy's location). With a coalescent source the
variant is a mutation on the source genealogy at its anchor (branch chosen with probability proportional to its
length, the reference leaf included, coded relative to the reference); with a file source each source haplotype
carries it independently with probability ``carrier_frequency``; explicit variants may list their carriers.

Meiosis (the released crossover draws are unchanged; structure only filters them and orders the products):

* a deletion carried by either homologue removes the crossovers inside it (no pairing partner, or no sequence);
* an inversion carried by exactly one homologue (a heterokaryotype) keeps the crossovers inside it only when they
  are even in number: a chromatid with an odd number of exchanges inside the loop is dicentric or acentric and gives
  no viable gamete, so the viable product is the one without them; a chromatid with an even number is balanced
  (two exchanges restore the orientation) and keeps them;
* the transmitted chromatid follows the physical order of the parent's chromosomes: a homokaryotypic inversion and a
  reversed segment are traversed backwards, a moved segment sits at its target, and an unlinked segment starts from
  its own independent homologue choice (a stream of its own, so no released draw moves). A single crossover inside a
  homokaryotypic inversion therefore switches the reference-axis haplotype three times (at start, at the crossover and
  at end), exactly as the reference-coordinate data show it.

Random streams (PCG64, SHA-256 keyed by stable identifiers after the realised seed): ``simulation.sv`` (seed, contig)
for the variants, carriers and paralogous differences; ``simulation.sv.gamete`` (seed, key...) for unlinked segments'
homologue choices; ``simulation.sv.reads`` (read seed, sample index) for the copy reads and deletion thinning. With the
model off nothing here runs and no stream is created.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..contracts.roles import freeze_record_arrays, record_field, schema_field
from ..core.errors import InputError, SettingsError
from . import parameters as p
from .haplotypes import Mosaic, source_at

__all__ = [
    "KINDS", "StructuralVariant", "StructureModel", "ContigStructure", "StructureTruth", "sv_stream",
    "draw_structure", "homologue_carriers", "filter_crossovers", "structured_transmit", "haplotype_carriers",
    "copy_alleles", "structural_counts", "StructureReads", "paralog_sites",
]

KINDS = ("inversion", "deletion", "duplication", "reversed", "moved", "unlinked")
INVERSION, DELETION, DUPLICATION, REVERSED, MOVED, UNLINKED = range(6)
POLYMORPHIC = (INVERSION, DELETION, DUPLICATION)
ORDER = (INVERSION, REVERSED, MOVED, UNLINKED)
STREAMS = ("simulation.sv", "simulation.sv.gamete", "simulation.sv.reads")


def sv_stream(name: str, seed: int, *key) -> np.random.Generator:
    """PCG64 stream keyed by the realised seed and a JSON key (SHA-256 entropy words), never by row order."""
    if name not in STREAMS:
        raise ValueError(f"unknown structure stream {name!r}")
    digest = hashlib.sha256(json.dumps([name, *key], ensure_ascii=True, separators=(",", ":")).encode()).digest()
    words = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, len(digest), 4)]
    return np.random.default_rng(np.random.SeedSequence([int(seed), *words]))


@dataclass(frozen=True)
class StructuralVariant:
    """One explicit variant: kind (KINDS), contig, half-open reference bp [start, end), target bp (duplication copy
    or moved segment location; ignored otherwise) and carriers (source haplotype indices; empty: drawn)."""

    kind: str = record_field("obs", dtype="str")
    contig: str = record_field("obs", dtype="str")
    start: int = record_field("obs", units="bp", dtype="int64")
    end: int = record_field("obs", units="bp", dtype="int64")
    target: int = record_field("obs", units="bp", dtype="int64", default=0)
    carriers: tuple[int, ...] = record_field("obs", dtype="tuple[int, ...]", default=())
    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise SettingsError(f"structural variant kind must be one of {', '.join(KINDS)}")
        if not isinstance(self.contig, str) or not self.contig:
            raise SettingsError("structural variant contig must be a nonempty identifier")
        for name in ("start", "end", "target"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise SettingsError(f"structural variant {name} must be a nonnegative integer bp coordinate")
        if not self.start < self.end:
            raise SettingsError("structural variant needs start < end")
        kind = KINDS.index(self.kind)
        if kind in (DUPLICATION, MOVED) and self.start < self.target < self.end:
            raise SettingsError("a duplication or moved segment's target must lie outside [start, end)")
        if kind in (MOVED,) and self.target in (self.start, self.end):
            raise SettingsError("a moved segment's target must differ from its own boundaries")
        carriers = tuple(int(c) for c in self.carriers)
        if any(c < 0 for c in carriers) or len(set(carriers)) != len(carriers):
            raise SettingsError("structural variant carriers must be distinct nonnegative source haplotype indices")
        if carriers and kind not in POLYMORPHIC:
            raise SettingsError("reference-misorder variants are carried by every haplotype; give no carriers")
        object.__setattr__(self, "carriers", carriers)


@dataclass(frozen=True)
class StructureModel:
    """Generating structural-variant parameters; numerical defaults are the ``cichlid`` preset (simulation/parameters.py
    states their sources). Disabled by default. Rates are expected counts (per contig, or per Mb of contig); lengths are
    means of exponential laws in bp, except large inversions, whose span is a uniform fraction of the contig."""

    enabled: bool = record_field("obs", dtype="bool", default=p.STRUCTURE_ENABLED)
    inversion_rate: float = record_field("obs", units="1/contig", dtype="float64", default=p.STRUCTURE_INVERSION_RATE)
    inversion_span_lo: float = record_field("obs", dtype="float64", default=p.STRUCTURE_INVERSION_SPAN_LO)
    inversion_span_hi: float = record_field("obs", dtype="float64", default=p.STRUCTURE_INVERSION_SPAN_HI)
    small_inversion_rate_per_mb: float = record_field(
        "obs", units="1/Mb", dtype="float64", default=p.STRUCTURE_SMALL_INVERSION_RATE_PER_MB
    )
    small_inversion_mean_bp: float = record_field(
        "obs", units="bp", dtype="float64", default=p.STRUCTURE_SMALL_INVERSION_MEAN_BP
    )
    deletion_rate_per_mb: float = record_field(
        "obs", units="1/Mb", dtype="float64", default=p.STRUCTURE_DELETION_RATE_PER_MB
    )
    deletion_mean_bp: float = record_field("obs", units="bp", dtype="float64", default=p.STRUCTURE_DELETION_MEAN_BP)
    duplication_rate_per_mb: float = record_field(
        "obs", units="1/Mb", dtype="float64", default=p.STRUCTURE_DUPLICATION_RATE_PER_MB
    )
    duplication_mean_bp: float = record_field(
        "obs", units="bp", dtype="float64", default=p.STRUCTURE_DUPLICATION_MEAN_BP
    )
    dispersed_fraction: float = record_field("obs", dtype="float64", default=p.STRUCTURE_DISPERSED_FRACTION)
    paralog_divergence: float = record_field("obs", dtype="float64", default=p.STRUCTURE_PARALOG_DIVERGENCE)
    misorder_rate_per_mb: float = record_field(
        "obs", units="1/Mb", dtype="float64", default=p.STRUCTURE_MISORDER_RATE_PER_MB
    )
    misorder_mean_bp: float = record_field("obs", units="bp", dtype="float64", default=p.STRUCTURE_MISORDER_MEAN_BP)
    carrier_frequency: float = record_field("obs", dtype="float64", default=p.STRUCTURE_CARRIER_FREQUENCY)
    variants: tuple[StructuralVariant, ...] = record_field("obs", dtype="tuple[StructuralVariant, ...]", default=())
    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise SettingsError("structure enabled must be a boolean")
        for name in ("inversion_rate", "small_inversion_rate_per_mb", "deletion_rate_per_mb", "duplication_rate_per_mb",
                     "misorder_rate_per_mb"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise SettingsError(f"structure {name} must be finite and nonnegative")
        for name in ("small_inversion_mean_bp", "deletion_mean_bp", "duplication_mean_bp", "misorder_mean_bp"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 1:
                raise SettingsError(f"structure {name} must be a finite length of at least 1 bp")
        for name in ("inversion_span_lo", "inversion_span_hi", "dispersed_fraction", "paralog_divergence",
                     "carrier_frequency"):
            if not 0 <= getattr(self, name) <= 1:
                raise SettingsError(f"structure {name} must be between zero and one")
        if not 0 < self.inversion_span_lo <= self.inversion_span_hi < 1:
            raise SettingsError("structure inversion spans need 0 < inversion_span_lo <= inversion_span_hi < 1")
        variants = tuple(v if isinstance(v, StructuralVariant) else StructuralVariant(*v) for v in self.variants)
        object.__setattr__(self, "variants", variants)
        for contig in {v.contig for v in variants}:
            _check_layout([KINDS.index(v.kind) for v in variants if v.contig == contig],
                          [v.start for v in variants if v.contig == contig],
                          [v.end for v in variants if v.contig == contig],
                          [v.target for v in variants if v.contig == contig], contig)

    @classmethod
    def preset(cls, name: str = p.STRUCTURE_PRESET) -> StructureModel:
        if name != p.STRUCTURE_PRESET:
            raise SettingsError(f"unknown structure preset {name!r}")
        return cls(enabled=True)


def _check_layout(kinds, starts, ends, targets, contig) -> None:
    """Order variants never overlap one another, content variants (deletion, duplication) never overlap one another;
    a duplication's copy never lands inside a deletion and a moved segment's target never inside another order
    variant. Content variants may lie inside order variants."""
    order = [(s, e) for k, s, e in zip(kinds, starts, ends) if k in ORDER]
    content = [(s, e) for k, s, e in zip(kinds, starts, ends) if k not in ORDER]
    for group, label in ((order, "inversions and misordered segments"), (content, "deletions and duplications")):
        group = sorted(group)
        if any(a[1] > b[0] for a, b in zip(group, group[1:])):
            raise SettingsError(f"{contig}: structural {label} must not overlap one another")
    deletions = [(s, e) for k, s, e in zip(kinds, starts, ends) if k == DELETION]
    for k, s, e, t in zip(kinds, starts, ends, targets):
        if k == DUPLICATION and any(a <= t < b for a, b in deletions):
            raise SettingsError(f"{contig}: a duplication's copy cannot lie inside a deletion")
        if k == MOVED and any(a < t < b for a, b in order):
            raise SettingsError(f"{contig}: a moved segment's target cannot lie inside an order variant")


@dataclass(frozen=True)
class ContigStructure:
    """Realised variants of one contig (sorted by start) and their carriers by stable source haplotype ID."""

    contig: str = record_field("obs", dtype="str")
    kind: NDArray[np.int8] = record_field("obs", axes=("V",), dtype="int8")
    start: NDArray[np.int64] = record_field("obs", axes=("V",), units="bp", dtype="int64")
    end: NDArray[np.int64] = record_field("obs", axes=("V",), units="bp", dtype="int64")
    target: NDArray[np.int64] = record_field("obs", axes=("V",), units="bp", dtype="int64")
    carriers: NDArray[np.bool_] = record_field("obs", axes=("S", "V"), dtype="bool")
    seed: int = record_field("obs", dtype="int")
    paralog_divergence: float = record_field("obs", dtype="float64")
    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        for name, dtype in (("kind", np.int8), ("start", np.int64), ("end", np.int64), ("target", np.int64)):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype))
        object.__setattr__(self, "carriers", np.asarray(self.carriers, dtype=bool))
        V = len(self.kind)
        if any(a.shape != (V,) for a in (self.start, self.end, self.target)) or self.carriers.ndim != 2 or (
            self.carriers.shape[1] != V
        ):
            raise InputError("contig structure axes do not agree")
        freeze_record_arrays(self)

    def __len__(self) -> int:
        return len(self.kind)

    @property
    def anchor(self) -> NDArray[np.int64]:
        """The reference bp whose source haplotype says who carries each variant."""
        return np.where(self.kind == DUPLICATION, self.target, self.start)

    @property
    def unlinked(self) -> NDArray[np.int64]:
        return np.flatnonzero(self.kind == UNLINKED)


@dataclass(frozen=True)
class StructureTruth:
    """Structure of one contig on the analysed axes (truth only).

    kind/start/end/target as ContigStructure; founder_carries bool [F2, V] (cross-founder haplotypes);
    sample_carries bool [N, 2, V] (analysed haplotypes; misorder variants are True for everyone);
    switch_offsets/switch_positions: per gamete of TruthCrossovers (same order), the reference-axis points where the
    transmitted haplotype changes parental homologue (bp); they equal the crossovers except around rearrangements;
    paralog_offsets/paralog_sites: per variant, the marker indices where a duplication's copy differs from its source.
    """

    kind: np.ndarray = record_field("obs", axes=("V",), units="1", dtype="int8")
    start: np.ndarray = record_field("obs", axes=("V",), units="bp", dtype="int64")
    end: np.ndarray = record_field("obs", axes=("V",), units="bp", dtype="int64")
    target: np.ndarray = record_field("obs", axes=("V",), units="bp", dtype="int64")
    founder_carries: np.ndarray = record_field("obs", axes=("F2", "V"), units="1", dtype="bool")
    sample_carries: np.ndarray = record_field("obs", axes=("N", "2", "V"), units="1", dtype="bool")
    switch_offsets: np.ndarray = record_field("obs", axes=("gametes+1",), units="index", dtype="int64")
    switch_positions: np.ndarray = record_field("obs", axes=("switches",), units="bp", dtype="float64")
    paralog_offsets: np.ndarray = record_field("obs", axes=("V+1",), units="index", dtype="int64")
    paralog_sites: np.ndarray = record_field("obs", axes=("paralog",), units="index", dtype="int64")
    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)

    def absent(self, positions: NDArray[np.int64]) -> NDArray[np.bool_]:
        """bool [N, L, 2]: analysed haplotype cells inside a deletion the haplotype carries."""
        positions = np.asarray(positions)
        out = np.zeros((self.sample_carries.shape[0], len(positions), 2), dtype=bool)
        for v in np.flatnonzero(self.kind == DELETION):
            lo, hi = np.searchsorted(positions, [self.start[v], self.end[v]])
            out[:, lo:hi, :] |= self.sample_carries[:, None, :, v]
        return out

    def founder_absent(self, positions: NDArray[np.int64]) -> NDArray[np.bool_]:
        """bool [F2, L]: cross-founder haplotype cells inside a deletion that founder haplotype carries."""
        positions = np.asarray(positions)
        out = np.zeros((self.founder_carries.shape[0], len(positions)), dtype=bool)
        for v in np.flatnonzero(self.kind == DELETION):
            lo, hi = np.searchsorted(positions, [self.start[v], self.end[v]])
            out[:, lo:hi] |= self.founder_carries[:, v][:, None]
        return out


# ---------------------------------------------------------------------------------------------------------------------
# Drawing a contig's variants and carriers
# ---------------------------------------------------------------------------------------------------------------------


def _random_variants(model: StructureModel, rng, first_bp: int, last_bp: int):
    """Candidate variants in a fixed order (large inversions, misorders, small inversions, deletions, duplications);
    a candidate overlapping an accepted one is rejected, so the accepted set is a deterministic function of the
    stream."""
    length = last_bp - first_bp + 1
    mb = length / 1e6
    kinds, starts, ends, targets = [], [], [], []

    def accept(kind, start, end, target):
        start, end = int(start), int(end)
        if not first_bp <= start < end <= last_bp + 1:
            return
        trial = (kinds + [kind], starts + [start], ends + [end], targets + [int(target)])
        try:
            _check_layout(*trial, "")
        except SettingsError:
            return
        kinds.append(kind); starts.append(start); ends.append(end); targets.append(int(target))

    for _ in range(int(rng.poisson(model.inversion_rate))):
        span = int(round(length * rng.uniform(model.inversion_span_lo, model.inversion_span_hi)))
        start = first_bp + int(rng.integers(0, max(1, length - span)))
        accept(INVERSION, start, start + span, 0)
    for _ in range(int(rng.poisson(model.misorder_rate_per_mb * mb))):
        size = max(1, int(math.ceil(rng.exponential(model.misorder_mean_bp))))
        start = first_bp + int(rng.integers(0, max(1, length - size)))
        kind = (REVERSED, MOVED, UNLINKED)[int(rng.integers(0, 3))]
        target = first_bp + int(rng.integers(0, length))
        accept(kind, start, start + size, target if kind == MOVED else 0)
    for _ in range(int(rng.poisson(model.small_inversion_rate_per_mb * mb))):
        size = max(1, int(math.ceil(rng.exponential(model.small_inversion_mean_bp))))
        start = first_bp + int(rng.integers(0, max(1, length - size)))
        accept(INVERSION, start, start + size, 0)
    for _ in range(int(rng.poisson(model.deletion_rate_per_mb * mb))):
        size = max(1, int(math.ceil(rng.exponential(model.deletion_mean_bp))))
        start = first_bp + int(rng.integers(0, max(1, length - size)))
        accept(DELETION, start, start + size, 0)
    for _ in range(int(rng.poisson(model.duplication_rate_per_mb * mb))):
        size = max(1, int(math.ceil(rng.exponential(model.duplication_mean_bp))))
        start = first_bp + int(rng.integers(0, max(1, length - size)))
        dispersed = rng.random() < model.dispersed_fraction
        target = first_bp + int(rng.integers(0, length)) if dispersed else start + size
        accept(DUPLICATION, start, start + size, target)
    return kinds, starts, ends, targets


def draw_structure(model: StructureModel, seed: int, contig: str, first_bp: int, last_bp: int, total_sources: int,
                   genealogy=None) -> ContigStructure:
    """Realise a contig's variants on the extent [first_bp, last_bp] and their carriers among ``total_sources`` stable
    source haplotypes. ``genealogy(anchors_bp, uniforms) -> bool [total_sources + 1, V]`` (the reference leaf last)
    gives derived states on the source genealogy; None: independent carriers with ``carrier_frequency``."""
    rng = sv_stream("simulation.sv", seed, contig)
    explicit = [v for v in model.variants if v.contig == contig]
    kinds = [KINDS.index(v.kind) for v in explicit]
    starts = [int(v.start) for v in explicit]
    ends = [int(v.end) for v in explicit]
    targets = [int(v.target) for v in explicit]
    for v in explicit:
        if not first_bp <= v.start < v.end <= last_bp + 1 or (KINDS.index(v.kind) in (DUPLICATION, MOVED) and not (
            first_bp <= v.target <= last_bp + 1
        )):
            raise SettingsError(f"{contig}: structural variant {v.kind} [{v.start}, {v.end}) lies outside the contig")
        if any(c >= total_sources for c in v.carriers):
            raise SettingsError(f"{contig}: structural variant carriers exceed the {total_sources} source haplotypes")
    if not explicit:
        kinds, starts, ends, targets = _random_variants(model, rng, first_bp, last_bp)
    order = np.argsort(np.asarray(starts, np.int64), kind="stable")
    kind = np.asarray(kinds, np.int8)[order] if kinds else np.empty(0, np.int8)
    start = np.asarray(starts, np.int64)[order] if starts else np.empty(0, np.int64)
    end = np.asarray(ends, np.int64)[order] if ends else np.empty(0, np.int64)
    target = np.asarray(targets, np.int64)[order] if targets else np.empty(0, np.int64)
    listed = [explicit[i].carriers for i in order] if explicit else [()] * len(kind)
    carriers = np.zeros((total_sources, len(kind)), dtype=bool)
    anchors = np.where(kind == DUPLICATION, target, start).astype(np.int64)
    uniforms = rng.random(len(kind))
    frequency = rng.random((total_sources, len(kind)))
    drawn = [v for v in range(len(kind)) if kind[v] in POLYMORPHIC and not listed[v]]
    if genealogy is not None and drawn:
        derived = np.asarray(genealogy(anchors[drawn], uniforms[drawn]), dtype=bool)
        reference = derived[-1]
        for column, v in enumerate(drawn):
            if kind[v] == INVERSION:
                carriers[:, v] = derived[:-1, column] ^ reference[column]
            else:
                # A deletion or duplication carried by the reference is sequence the other haplotypes lack or the
                # reference holds: neither changes reference-axis calls, so only non-reference carriers count.
                carriers[:, v] = derived[:-1, column] & ~reference[column]
    else:
        for v in drawn:
            carriers[:, v] = frequency[:, v] < model.carrier_frequency
    for v, listed_carriers in enumerate(listed):
        if listed_carriers:
            carriers[list(listed_carriers), v] = True
        if kind[v] not in POLYMORPHIC:
            carriers[:, v] = True
    return ContigStructure(contig, kind, start, end, target, carriers, int(seed), float(model.paralog_divergence))


# ---------------------------------------------------------------------------------------------------------------------
# Meiosis
# ---------------------------------------------------------------------------------------------------------------------


def homologue_carriers(mosaic: Mosaic, structure: ContigStructure) -> NDArray[np.bool_]:
    """bool [V]: whether one haplotype (a Mosaic of stable source IDs) carries each variant, read at the anchors."""
    if not len(structure):
        return np.zeros(0, dtype=bool)
    sources = source_at(mosaic, structure.anchor.astype(np.float64))
    if np.any(sources < 0) or np.any(sources >= structure.carriers.shape[0]):
        raise InputError("a haplotype names a source outside the structure's carrier table")
    return structure.carriers[sources, np.arange(len(structure))]


def filter_crossovers(events, structure: ContigStructure, states: NDArray[np.bool_]) -> NDArray[np.float64]:
    """The released crossovers a viable gamete keeps, given both homologues' carriers ``states`` bool [2, V].

    A crossover at c lies inside [start, end) when start < c < end (c switches positions >= c). Deletions carried by
    either homologue drop every crossover inside; heterokaryotypic inversions drop an odd number of inside crossovers.
    """
    events = np.asarray(events, dtype=np.float64)
    if not len(events) or not len(structure):
        return events
    keep = np.ones(len(events), dtype=bool)
    for v in range(len(structure)):
        inside = (events > structure.start[v]) & (events < structure.end[v])
        if not inside.any():
            continue
        if structure.kind[v] == DELETION and (states[0, v] or states[1, v]):
            keep &= ~inside
    for v in range(len(structure)):
        if structure.kind[v] == INVERSION and states[0, v] != states[1, v]:
            inside = keep & (events > structure.start[v]) & (events < structure.end[v])
            if inside.sum() % 2:
                keep &= ~inside
    return events[keep]


def _blocks(structure: ContigStructure, states: NDArray[np.bool_]):
    """Physical layout of the parent's chromosomes as blocks of the reference axis.

    Returns (lo float64[B] sorted, component int64[B], rank int64[B], reversed bool[B]); block b covers
    [lo[b], lo[b + 1]) and the first block starts at -inf. Component 0 is the contig; unlinked segment k is component
    k + 1 (its own chromosome)."""
    cuts = {-np.inf}
    for v in range(len(structure)):
        k = structure.kind[v]
        if k in ORDER:
            cuts.update((float(structure.start[v]), float(structure.end[v])))
        if k == MOVED:
            cuts.add(float(structure.target[v]))
    lo = np.asarray(sorted(cuts), dtype=np.float64)
    hi = np.r_[lo[1:], np.inf]
    component = np.zeros(len(lo), np.int64)
    reverse = np.zeros(len(lo), bool)
    anchor = lo.copy()
    unlinked = 0
    for v in range(len(structure)):
        k = structure.kind[v]
        if k not in ORDER:
            continue
        inside = (lo >= structure.start[v]) & (hi <= structure.end[v])
        if k == INVERSION:
            reverse[inside] = bool(states[0, v] and states[1, v])
        elif k == REVERSED:
            reverse[inside] = True
        elif k == MOVED:
            anchor[inside] = structure.target[v] - 0.5
        else:
            unlinked += 1
            component[inside] = unlinked
    rank = np.empty(len(lo), np.int64)
    for c in np.unique(component):
        members = np.flatnonzero(component == c)
        order = members[np.lexsort((lo[members], anchor[members]))]
        rank[order] = np.arange(len(order))
    return lo, component, rank, reverse


def _owners(points, cuts, layout, first, coins):
    """Homologue (0/1) of each reference point: the component's starting homologue plus the number of crossover cuts
    physically before the point, mod 2. Within a forward block a cut at c precedes points >= c; within a reversed block
    it precedes points < c (the chromosome is traversed backwards there)."""
    lo, component, rank, reverse = layout
    points = np.asarray(points, np.float64)
    cuts = np.asarray(cuts, np.float64)
    bp = np.searchsorted(lo, points, side="right") - 1
    start = np.where(component[bp] == 0, first, np.asarray(coins, np.int64)[np.maximum(component[bp] - 1, 0)]
                     if len(coins) else first)
    if not len(cuts):
        return start % 2
    bc = np.searchsorted(lo, cuts, side="right") - 1
    off_p = np.where(reverse[bp], -points, points)
    off_c = np.where(reverse[bc], -cuts, cuts)
    tie_c = np.where(reverse[bc], 2, 0)
    same = component[bp][:, None] == component[bc][None, :]
    rp, rc = rank[bp][:, None], rank[bc][None, :]
    op, oc = off_p[:, None], off_c[None, :]
    before = same & ((rc < rp) | ((rc == rp) & ((oc < op) | ((oc == op) & (tie_c[None, :] < 1)))))
    return (start + before.sum(axis=1)) % 2


def structured_transmit(parent, first: int, crossovers, new_flips, positions, structure: ContigStructure, coins=()):
    """transmit() under structure: returns (Mosaic, kept crossovers float64, reference-axis switches float64).

    ``crossovers``: the released draw (bp, sorted); ``coins``: one starting homologue per unlinked variant."""
    states = np.stack([homologue_carriers(parent[0], structure), homologue_carriers(parent[1], structure)])
    kept = filter_crossovers(crossovers, structure, states)
    layout = _blocks(structure, states)
    extent_start = float(min(parent[0].starts[0], parent[1].starts[0]))
    interior = layout[0][(layout[0] > extent_start) & np.isfinite(layout[0])]
    boundaries = np.unique(np.concatenate((parent[0].starts, parent[1].starts, kept, interior)))
    boundaries = boundaries[boundaries >= extent_start]
    owner = _owners(boundaries, kept, layout, first, coins)
    sources = np.empty(len(boundaries), dtype=np.int32)
    founders = np.empty(len(boundaries), dtype=np.int32)
    inherited = []
    for h, mosaic in enumerate(parent):
        mask = owner == h
        index = np.searchsorted(mosaic.starts, boundaries[mask], side="right") - 1
        sources[mask], founders[mask] = mosaic.source[index], mosaic.founder[index]
        if len(mosaic.flips):
            if positions is None:
                raise ValueError("marker positions are required to transmit flips")
            marker = np.asarray(positions, dtype=np.float64)[mosaic.flips]
            inherited.append(mosaic.flips[_owners(marker, kept, layout, first, coins) == h])
    switches = boundaries[1:][owner[1:] != owner[:-1]]
    keep = np.r_[True, (sources[1:] != sources[:-1]) | (founders[1:] != founders[:-1])]
    flips = np.concatenate(inherited) if inherited else np.empty(0, dtype=np.int64)
    flips = np.setxor1d(flips, np.asarray(new_flips, dtype=np.int64))
    return Mosaic(boundaries[keep], sources[keep], founders[keep], flips), kept, switches.astype(np.float64)


# ---------------------------------------------------------------------------------------------------------------------
# Reads and truth
# ---------------------------------------------------------------------------------------------------------------------


def haplotype_carriers(haplotypes, structure: ContigStructure) -> NDArray[np.bool_]:
    """bool [N, 2, V] for [N][2] Mosaic pairs."""
    return np.asarray([[homologue_carriers(h, structure) for h in pair] for pair in haplotypes], dtype=bool).reshape(
        len(haplotypes), 2, len(structure)
    )


def paralog_sites(structure: ContigStructure, positions: NDArray[np.int64]):
    """Per duplication, the marker indices inside [start, end) where the copy differs from its source (paralogous
    sequence variants), each site independently with probability paralog_divergence. Returns (offsets [V+1],
    sites)."""
    rng = sv_stream("simulation.sv", structure.seed, structure.contig, "paralog")
    offsets, parts = [0], []
    for v in range(len(structure)):
        if structure.kind[v] == DUPLICATION:
            lo, hi = np.searchsorted(positions, [structure.start[v], structure.end[v]])
            flip = rng.random(hi - lo) < structure.paralog_divergence
            parts.append(lo + np.flatnonzero(flip))
        else:
            parts.append(np.empty(0, np.int64))
        offsets.append(offsets[-1] + len(parts[-1]))
    return np.asarray(offsets, np.int64), (np.concatenate(parts).astype(np.int64) if parts else np.empty(0, np.int64))


def copy_alleles(offspring, structure: ContigStructure, positions, panel_alleles, panel_ids, paralog):
    """(present bool [N, 2, L], copies int8 [N, 2, L]) of the analysed haplotypes.

    present: False inside a deletion the haplotype carries. copies: -1, or the allele (REF/ALT coding) of the extra copy
    the haplotype carries at that marker: the source haplotype at the copy's location, read at [start, end), with the
    paralogous sites complemented."""
    N, L = len(offspring), len(positions)
    present = np.ones((N, 2, L), dtype=bool)
    copies = np.full((N, 2, L), -1, dtype=np.int8)
    offsets, sites = paralog
    carries = haplotype_carriers(offspring, structure)
    for v in range(len(structure)):
        lo, hi = np.searchsorted(positions, [structure.start[v], structure.end[v]])
        if hi <= lo:
            continue
        if structure.kind[v] == DELETION:
            present[:, :, lo:hi] &= ~carries[:, :, v, None]
        elif structure.kind[v] == DUPLICATION:
            flips = np.zeros(hi - lo, dtype=np.int8)
            flips[sites[offsets[v]:offsets[v + 1]] - lo] = 1
            for i, pair in enumerate(offspring):
                for h, mosaic in enumerate(pair):
                    if not carries[i, h, v]:
                        continue
                    source = int(source_at(mosaic, np.array([float(structure.target[v])]))[0])
                    row = np.searchsorted(panel_ids, source)
                    if row >= len(panel_ids) or panel_ids[row] != source:
                        # A source carried only between markers has no panel row: the copy then takes the source the
                        # homologue carries at the segment's first marker (always a panel row).
                        source = int(source_at(mosaic, np.array([float(positions[lo])]))[0])
                        row = np.searchsorted(panel_ids, source)
                        if row >= len(panel_ids) or panel_ids[row] != source:
                            raise InputError("a duplication's copy names a source absent from the panel")
                    copies[i, h, lo:hi] = panel_alleles[row, lo:hi] ^ flips
    return present, copies


@dataclass(frozen=True)
class StructureReads:
    """What the read samplers need: present bool [N, 2, L], copies int8 [N, 2, L] (copy_alleles) and the paralogous
    sites (paralog_sites)."""

    present: np.ndarray = record_field("obs", axes=("N", "2", "L"), dtype="bool")
    copies: np.ndarray = record_field("obs", axes=("N", "2", "L"), dtype="int8")
    paralog_offsets: np.ndarray = record_field("obs", axes=("V+1",), dtype="int64")
    paralog_sites: np.ndarray = record_field("obs", axes=("paralog",), dtype="int64")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


def structural_counts(counts, alleles, present, copies, lam, error, rng, thin):
    """Apply deletions and extra copies to one sample's sampled counts int64 [L, 2] (returned new).

    alleles int8 [2, L] (the homologues' sequence alleles), present bool [2, L], copies int8 [2, L], lam float64 [L]
    (expected reads of one homologue copy at each marker), error (probability that a read reports the other allele).
    thin: the sampler did not know about deletions, so each sampled read is assigned to a homologue (1/2 each) and
    reads of an absent homologue are removed, the kept reads redrawn per homologue; otherwise the counts already
    exclude absent homologues. Each carried copy adds Poisson(lam) reads of its allele at error ``error``."""
    out = np.array(counts, dtype=np.int64, copy=True)
    if thin:
        cells = np.flatnonzero(~present.all(axis=0))
        if len(cells):
            total = out[cells].sum(axis=1)
            first = rng.binomial(total, 0.5)
            reads = np.stack((first, total - first))
            reads = np.where(present[:, cells], reads, 0)
            p_alt = np.where(alleles[:, cells] == 1, 1 - error, error)
            alt = rng.binomial(reads, p_alt).sum(axis=0)
            kept = reads.sum(axis=0)
            out[cells, 0], out[cells, 1] = kept - alt, alt
    for h in range(2):
        cells = np.flatnonzero(copies[h] >= 0)
        if len(cells):
            extra = rng.poisson(lam[cells])
            alt = rng.binomial(extra, np.where(copies[h, cells] == 1, 1 - error, error))
            out[cells, 0] += extra - alt
            out[cells, 1] += alt
    return out
