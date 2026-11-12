"""Records of a simulation: its manifest, each contig's simulated reads and its truth.

Inference never reads the truth. Records (prefix ``simulation``, an input step, not a stage):

* run ``simulation/manifest``: :class:`SimulationManifest` (state, seeds, contigs, analysed samples,
  the pedigree of the analysed samples and of every generated individual);
* contig ``simulation/reads``: positions int64[L], counts int32[N, L, 2] of the analysed samples;
* contig ``simulation/truth``: :class:`ContigTruth`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts.roles import freeze_record_arrays, record_field, require_schema, schema_field
from ..core.errors import InputError
from .pedigree import PedigreeTable

MANIFEST = "simulation/manifest"
READS = "simulation/reads"
TRUTH = "simulation/truth"
SCHEMA = 2
IN_PROGRESS, COMPLETE = "in_progress", "complete"


def require_simulation_schema(record) -> None:
    """Refuse old simulation products with the release's regeneration instruction."""
    if record.get("schema") == 1:
        raise InputError(
            "simulation schema 1 is unsupported by release 4.15; regenerate the simulation in a new output root"
        )
    require_schema(record, SCHEMA)


@dataclass(frozen=True)
class SimulationManifest:
    """The run-level record of a simulation.

    state: "in_progress" (the realised seed is bound, contigs are being simulated) or "complete".
    requested_seed: the configured seed (None = random); seed: the realised seed. sample_ids: the
    analysed (sequenced) individuals in pedigree order; observed: their pedigree rows (the truth
    evaluation compares with); generated: every generated individual. identity: the source
    identity it was made for.
    """

    state: str = record_field("state", axes=(), units="1", dtype="str")
    requested_seed: int | None = record_field("obs", axes=(), units="1", dtype="int | None")
    seed: int = record_field("obs", axes=(), units="1", dtype="int")
    contigs: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    identity: dict = record_field("diagnostic", axes=(), units="1", dtype="mapping")
    sample_ids: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]", default=())
    observed: PedigreeTable | None = record_field("obs", axes=(), units="1", dtype="PedigreeTable | None", default=None)
    generated: PedigreeTable | None = record_field(
        "obs", axes=(), units="1", dtype="PedigreeTable | None", default=None
    )

    stocks: tuple = record_field("obs", units="1", dtype="tuple", default=())
    populations: tuple[str, ...] = record_field("obs", units="1", dtype="tuple[str, ...]", default=())

    def record(self) -> dict:
        return dict(
            schema=SCHEMA,
            state=self.state,
            requested_seed=self.requested_seed,
            seed=self.seed,
            contigs=list(self.contigs),
            identity=self.identity,
            sample_ids=list(self.sample_ids),
            observed=None if self.observed is None else self.observed.record(),
            generated=None if self.generated is None else self.generated.record(),
            stocks=self.stocks,
            populations=self.populations,
        )

    @classmethod
    def from_record(cls, record) -> SimulationManifest:
        require_simulation_schema(record)
        return cls(
            state=record["state"],
            requested_seed=record["requested_seed"],
            seed=record["seed"],
            contigs=tuple(record["contigs"]),
            identity=record["identity"],
            sample_ids=tuple(record["sample_ids"]),
            observed=None if record["observed"] is None else PedigreeTable.from_record(record["observed"]),
            generated=None if record["generated"] is None else PedigreeTable.from_record(record["generated"]),
            stocks=tuple(record["stocks"]),
            populations=tuple(record["populations"]),
        )

    schema: int = schema_field(SCHEMA)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class TruthPainting:
    """Diploid labels on half-open marker-index intervals [start, end).

    Sample s owns chunks offsets[s]:offsets[s+1]. hap1 and hap2 are
    cross-founder IDs (painting) or stable source IDs (source_painting).
    """

    offsets: np.ndarray = record_field("obs", axes=("N+1",), units="index", dtype="int64")
    start: np.ndarray = record_field("obs", axes=("chunks",), units="index", dtype="int64")
    end: np.ndarray = record_field("obs", axes=("chunks",), units="index", dtype="int64")
    hap1: np.ndarray = record_field("obs", axes=("chunks",), units="index", dtype="int32")
    hap2: np.ndarray = record_field("obs", axes=("chunks",), units="index", dtype="int32")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class TruthCrossovers:
    """One row per gamete of an analysed child: the crossovers its parent's meiosis made.

    child_index: row of the child on the analysed-sample axis; child, parent: names; parent_slot:
    0 = the child's first (maternal) haplotype, 1 = second; cohort_index: 0 = children of the
    founders. Gamete g's crossover positions (bp, float) are positions[offsets[g]:offsets[g + 1]].
    """

    child_index: np.ndarray = record_field("obs", axes=("gametes",), units="index", dtype="int64")
    child: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    parent: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    parent_slot: np.ndarray = record_field("obs", axes=("gametes",), units="index", dtype="int8")
    cohort_index: np.ndarray = record_field("obs", axes=("gametes",), units="index", dtype="int64")
    offsets: np.ndarray = record_field("obs", axes=("gametes+1",), units="index", dtype="int64")
    positions: np.ndarray = record_field("obs", axes=("crossovers",), units="bp", dtype="float64")

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class FounderMosaics:
    """Source segments per cross-founder haplotype, on the marker-index axis."""

    offsets: np.ndarray = record_field("obs", axes=("F2+1",), units="index", dtype="int64")
    start: np.ndarray = record_field("obs", axes=("segments",), units="index", dtype="int64")
    source: np.ndarray = record_field("obs", axes=("segments",), units="index", dtype="int32")

    def __post_init__(self):
        freeze_record_arrays(self)


@dataclass(frozen=True)
class TruthFlips:
    """Sorted unique complemented markers per analysed haplotype, sample then slot."""

    offsets: np.ndarray = record_field("obs", axes=("2N+1",), units="index", dtype="int64")
    markers: np.ndarray = record_field("obs", axes=("flips",), units="index", dtype="int64")

    def __post_init__(self):
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ContigTruth:
    """Truth v2 on the full site axis; founder alleles are reconstructed on demand."""

    sample_ids: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    positions: np.ndarray = record_field("obs", axes=("L",), units="bp", dtype="int64")
    alleles: np.ndarray = record_field("obs", axes=("N", "L", "2"), units="1", dtype="int8")
    painting: TruthPainting = record_field("obs", axes=(), units="1", dtype="TruthPainting")
    crossovers: TruthCrossovers = record_field("obs", axes=(), units="1", dtype="TruthCrossovers")
    source_haplotypes: np.ndarray = record_field("obs", axes=("S", "L"), units="1", dtype="int8")
    source_id: np.ndarray = record_field("obs", axes=("S",), units="index", dtype="int32")
    source_population: np.ndarray = record_field("obs", axes=("S",), units="index", dtype="int32")
    total_sources: int = record_field("obs", units="1", dtype="int")
    reference_derived: np.ndarray = record_field("obs", axes=("L",), units="1", dtype="bool")
    founder_mosaics: FounderMosaics = record_field("obs", units="1", dtype="FounderMosaics")
    source_painting: TruthPainting = record_field("obs", units="1", dtype="TruthPainting")
    flips: TruthFlips = record_field("obs", units="1", dtype="TruthFlips")
    dropped_positions: np.ndarray = record_field("obs", axes=("D",), units="bp", dtype="int64")
    missed: np.ndarray | None = record_field("obs", axes=("N", "L"), units="1", dtype="bool | None", default=None)
    block: np.ndarray | None = record_field("obs", axes=("N", "L"), units="1", dtype="bool | None", default=None)
    systematic_site: np.ndarray | None = record_field("obs", axes=("L",), units="1", dtype="bool | None", default=None)
    spurious: np.ndarray | None = record_field("obs", axes=("L",), units="1", dtype="bool | None", default=None)
    schema: int = schema_field(SCHEMA)

    def __post_init__(self) -> None:
        mark = np.zeros(len(self.positions), bool) if self.spurious is None else np.asarray(self.spurious, bool)
        if mark.shape != self.positions.shape or np.any(self.alleles[:, mark]) or np.any(self.source_haplotypes[:, mark]):
            raise InputError("spurious truth sites must be aligned and carry no ALT allele")
        object.__setattr__(self, "spurious", mark)
        freeze_record_arrays(self)

    @property
    def founder_haplotypes(self) -> np.ndarray:
        """Computed compatibility view for released evaluation consumers; never stored."""
        return founder_alleles(self)


def founder_alleles(truth: ContigTruth) -> np.ndarray:
    """Rebuild the cross-founder allele axis using stable source IDs, not row IDs."""
    mosaics = truth.founder_mosaics
    result = np.empty((len(mosaics.offsets) - 1, len(truth.positions)), dtype=np.int8)
    for h, (first, stop) in enumerate(zip(mosaics.offsets[:-1], mosaics.offsets[1:], strict=True)):
        for i in range(first, stop):
            left = mosaics.start[i]
            right = mosaics.start[i + 1] if i + 1 < stop else len(truth.positions)
            row = np.searchsorted(truth.source_id, mosaics.source[i])
            if row == len(truth.source_id) or truth.source_id[row] != mosaics.source[i]:
                raise InputError("founder mosaic names a source absent from the truth panel")
            result[h, left:right] = truth.source_haplotypes[row, left:right]
    return result


def truth_painting(ancestry, positions, *, source=False) -> TruthPainting:
    """Combine Mosaic pairs into exact half-open marker-index chunks."""
    offsets, starts, ends, hap1s, hap2s = [0], [], [], [], []
    for pair in ancestry:
        boundaries = np.unique(
            np.r_[
                0,
                np.searchsorted(positions, pair[0].starts),
                np.searchsorted(positions, pair[1].starts),
                len(positions),
            ]
        )
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            labels = []
            for mosaic in pair:
                segment = np.searchsorted(mosaic.starts, positions[left], side="right") - 1
                labels.append((mosaic.source if source else mosaic.founder)[segment])
            if len(starts) > offsets[-1] and labels == [hap1s[-1], hap2s[-1]]:
                ends[-1] = right
            else:
                starts.append(left)
                ends.append(right)
                hap1s.append(labels[0])
                hap2s.append(labels[1])
        offsets.append(len(starts))
    return TruthPainting(
        np.asarray(offsets, np.int64),
        np.asarray(starts, np.int64),
        np.asarray(ends, np.int64),
        np.asarray(hap1s, np.int32),
        np.asarray(hap2s, np.int32),
    )


def truth_founder_mosaics(founders, positions) -> FounderMosaics:
    """Source segments retained at markers, in cross-founder haplotype order."""
    offsets, starts, sources = [0], [], []
    for pair in founders:
        for mosaic in pair:
            boundaries = np.unique(np.r_[0, np.searchsorted(positions, mosaic.starts), len(positions)])
            for left in boundaries[:-1]:
                index = np.searchsorted(mosaic.starts, positions[left], side="right") - 1
                source = mosaic.source[index]
                if len(starts) == offsets[-1] or sources[-1] != source:
                    starts.append(left)
                    sources.append(source)
            offsets.append(len(starts))
    return FounderMosaics(np.asarray(offsets, np.int64), np.asarray(starts, np.int64), np.asarray(sources, np.int32))


def truth_flips(offspring) -> TruthFlips:
    """Pack analysed haplotype flips without gathering any alleles."""
    parts = [mosaic.flips for pair in offspring for mosaic in pair]
    return TruthFlips(
        np.r_[0, np.cumsum([len(part) for part in parts])].astype(np.int64),
        np.concatenate(parts).astype(np.int64) if parts else np.empty(0, np.int64),
    )


def truth_crossovers(gametes, child_rows, child_names, cohort_of_row) -> TruthCrossovers:
    """Gametes (child row, slot, parent, positions) of analysed children, on the analysed axis.

    child_rows: {generated row: analysed index}; rows not analysed are left out.
    """
    rows = [
        (child_rows[row], child_names[row], parent, slot, cohort_of_row[row] - 1, positions)
        for row, slot, parent, positions in gametes
        if row in child_rows
    ]
    lengths = [len(row[5]) for row in rows]
    return TruthCrossovers(
        child_index=np.asarray([row[0] for row in rows], np.int64),
        child=tuple(row[1] for row in rows),
        parent=tuple(row[2] for row in rows),
        parent_slot=np.asarray([row[3] for row in rows], np.int8),
        cohort_index=np.asarray([row[4] for row in rows], np.int64),
        offsets=np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
        positions=(np.concatenate([row[5] for row in rows]).astype(np.float64) if rows else np.empty(0, np.float64)),
    )


def truth_record(truth: ContigTruth) -> dict:
    """Encode truth v2 without storing the computed cross-founder allele array."""
    result = dict(truth.__dict__)
    for name in ("painting", "source_painting", "founder_mosaics", "flips", "crossovers"):
        result[name] = getattr(truth, name).__dict__
    return result


def truth_from_record(payload: dict) -> ContigTruth:
    """Decode truth v2 after the boundary reader has checked publication identity."""
    require_simulation_schema(payload)
    values = dict(payload)
    values["sample_ids"] = tuple(values["sample_ids"])
    for name, cls in (
        ("painting", TruthPainting),
        ("source_painting", TruthPainting),
        ("founder_mosaics", FounderMosaics),
        ("flips", TruthFlips),
        ("crossovers", TruthCrossovers),
    ):
        values[name] = cls(**values[name])
    return ContigTruth(**values)
