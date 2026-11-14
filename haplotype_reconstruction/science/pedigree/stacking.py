"""Sum every informative contig's pieces into the genome's material units (DESIGN_417 D1, I1, D7).

Owns PD02's score-record gather. Records arrive in configured contig order, each holding its pieces of the
genome units (blocks.pieces) and its offset on the genome axis; the stacking checks that the offsets chain
exactly into U, adds each piece into its unit in genome order (every piece field is a sum of per-bin increments,
so a unit's sum equals the direct per-bin sum), drops empty nominal units, and spreads the genome M1 floor
deficit of the screened-out identities over all units by bins (D7): unit floor = unit M0 + (sum over contigs of
the screen's floors - genome M0) x unit bins / genome bins, so the floors sum per child to the screen's genome
cutoff. Retained pairs keep their own per-unit M1. One record is held at a time.
Method: docs/methods/pedigree.md#material-units.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ...contracts.pedigree import (
    CandidatePanel,
    ChromosomeScores,
    ChromosomeSummaryView,
    GenomeScores,
    ParentEligibility,
)
from ...core.errors import InputError
from ..shared.material import unit_count
from .parameters import PedigreeParameters
from .regimes import MaterialAxis, classify

__all__ = ["stack"]

_PIECE = (
    "zero", "two", "edge_matched", "edge_exposed", "pair_explained", "pair_exposed", "junctions", "callable_bins",
    "transmission_log_likelihood", "total_bins", "one_raw",
)


def stack(
    sample_ids: tuple[str, ...],
    panel: CandidatePanel,
    chromosomes: Iterable[ChromosomeScores],
    eligibility: ParentEligibility,
    parameters: PedigreeParameters,
    *,
    axis: MaterialAxis | None = None,
    summary: tuple[ChromosomeSummaryView, ...] = (),
) -> GenomeScores:
    """Own C-contiguous unit arrays; records remain borrowed and unchanged.

    ``axis``: the material axis of the prepared layouts (regimes.axis); when given, the records must reproduce its
    contigs, offsets and U exactly, and the unit arrays are filled while records stream. Without it the records are
    read first (tests and replays)."""
    if sample_ids != eligibility.sample_ids:
        raise InputError("score stacking eligibility differs from the sample axis")
    material = parameters.material
    if axis is None:
        chromosomes = tuple(chromosomes)
        if not chromosomes:
            raise InputError("pedigree scores require at least one informative contig")
        end = chromosomes[-1].end_morgans
        axis = MaterialAxis(tuple(c.contig for c in chromosomes), tuple(c.offset_morgans for c in chromosomes),
                            tuple(c.end_morgans for c in chromosomes), float(end),
                            unit_count(float(end), material.unit_morgans, material.tail_morgans))
    n = len(sample_ids)
    edges = np.ascontiguousarray(np.argwhere(eligibility.parents), dtype=np.int64)
    keys = edges[:, 0] * n + edges[:, 1]
    count = axis.units
    sums: dict[str, np.ndarray] = {}
    occupied = np.zeros(count, dtype=np.bool_)
    working = np.zeros(count)
    markers = np.zeros(count, dtype=np.int64)
    floors = np.zeros(n)
    contigs: list[str] = []
    pieces, piece_contig, piece_unit, piece_bins, piece_markers = [], [], [], [], []
    contig_junctions, contig_callable, contig_bins, contig_working, contig_markers = [], [], [], [], []
    dyads = retained = None
    for index, value in enumerate(chromosomes):
        if index >= len(axis.contigs) or value.contig != axis.contigs[index]:
            raise InputError("score records do not follow the informative contig axis")
        if value.offset_morgans != axis.offsets[index] or value.end_morgans != axis.ends[index]:
            raise InputError(f"{value.contig}: score record offset differs from the genome material axis")
        b = value.pieces
        if dyads is None:
            dyads = np.array(value.junction_dyads, dtype=np.int64, order="C", copy=True)
            retained = np.array(b.retained_pairs, dtype=np.int64, copy=True)
        elif not np.array_equal(dyads, value.junction_dyads):
            raise InputError("chromosome scores orient different dyads")
        elif not np.array_equal(retained, b.retained_pairs):
            raise InputError("chromosome scores retain different pairs")
        if (b.zero.shape[1:] != (n,) or b.two.shape[1:] != (len(panel.trios),)
                or b.edge_exposed.shape[1:] != (len(edges),)):
            raise InputError("chromosome score sample/panel/edge axis differs")
        nominal = np.asarray([d.nominal_index for d in b.descriptors], dtype=np.int64)
        if len(nominal) and (nominal.min() < 0 or nominal.max() >= count or np.any(np.diff(nominal) < 0)):
            raise InputError(f"{value.contig}: score pieces lie outside the genome's units")
        if not sums:
            sums = {name: np.zeros((count, *np.shape(getattr(b, name))[1:])) for name in _PIECE}
        for row, unit in enumerate(nominal):
            for name in _PIECE:
                sums[name][unit] += getattr(b, name)[row]
            working[unit] += b.descriptors[row].working_morgans
            markers[unit] += int(b.informative_markers[row])
        occupied[nominal] = True
        floors += value.floor
        contigs.append(value.contig)
        pieces.extend(b.descriptors)
        piece_contig.extend([index] * len(nominal))
        piece_unit.extend(nominal.tolist())
        piece_bins.extend(np.asarray(b.total_bins, dtype=np.int64).tolist())
        piece_markers.extend(np.asarray(b.informative_markers, dtype=np.int64).tolist())
        contig_junctions.append(np.asarray(b.junctions).sum(axis=0))
        contig_callable.append(np.asarray(b.callable_bins).sum(axis=0))
        contig_bins.append(float(np.asarray(b.total_bins).sum()))
        contig_working.append(float(value.working_morgans))
        contig_markers.append(int(value.informative_markers))
        del value, b
    if tuple(contigs) != axis.contigs:
        raise InputError("pedigree scores require every informative contig of the material axis")
    if not occupied.any():
        raise InputError("pedigree scores require occupied material units")
    rows = np.flatnonzero(occupied)
    dense = np.full(count, -1, dtype=np.int64)
    dense[rows] = np.arange(len(rows))
    unit = {name: np.ascontiguousarray(value[rows], dtype=np.float64) for name, value in sums.items()}
    # D7: the genome deficit of the screened-out M1 floors, spread over every unit by its bins.
    bins = unit["total_bins"]
    deficit = floors - unit["zero"].sum(axis=0)
    floor = unit["zero"] + (bins / bins.sum())[:, None] * deficit[None]
    one = np.ascontiguousarray(floor[:, edges[:, 0]])
    if retained is not None and len(retained):
        wanted = retained[:, 0] * n + retained[:, 1]
        columns = np.searchsorted(keys, wanted)
        if np.any(columns >= len(keys)) or np.any(keys[np.minimum(columns, len(keys) - 1)] != wanted):
            raise InputError("retained pairs lie outside the eligible edges")
        one[:, columns] = unit["one_raw"]
    morgans = axis.morgans
    regime = classify(morgans, len(rows), parameters).name
    return GenomeScores(
        sample_ids,
        tuple(f"unit:{int(u)}" for u in rows),
        panel,
        edges,
        zero=unit["zero"],
        one=one,
        two=unit["two"],
        edge_matched=unit["edge_matched"],
        edge_exposed=unit["edge_exposed"],
        pair_explained=unit["pair_explained"],
        pair_exposed=unit["pair_exposed"],
        total_bins=np.ascontiguousarray(bins),
        junctions=unit["junctions"],
        callable_bins=unit["callable_bins"],
        informative_markers=np.ascontiguousarray(markers[rows]),
        junction_dyads=dyads,
        transmission_log_likelihood=unit["transmission_log_likelihood"],
        working_morgans=np.ascontiguousarray(working[rows]),
        material_morgans=float(morgans),
        regime=regime,
        real_contigs=tuple(contigs),
        pieces=tuple(pieces),
        piece_contig=np.asarray(piece_contig, dtype=np.int64),
        piece_unit=dense[np.asarray(piece_unit, dtype=np.int64)],
        piece_bins=np.asarray(piece_bins, dtype=np.int64),
        piece_markers=np.asarray(piece_markers, dtype=np.int64),
        contig_junctions=np.ascontiguousarray(np.stack(contig_junctions)),
        contig_callable_bins=np.ascontiguousarray(np.stack(contig_callable)),
        contig_total_bins=np.asarray(contig_bins),
        contig_working_morgans=np.asarray(contig_working),
        contig_markers=np.asarray(contig_markers, dtype=np.int64),
        chromosome_summary=tuple(summary),
    )
