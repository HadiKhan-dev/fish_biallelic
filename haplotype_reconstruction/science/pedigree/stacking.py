"""Stack once-floored chromosome scores while retaining only eligible dense edges.

Owns PD02's score-record gather; never floors M1 a second time or scores a model.
Each input may be decoded lazily; retained copies release its dense N×N buffers
before the next chromosome. Output contig order is the iterable's declared order.
Method: docs/methods/pedigree.md#configuration-likelihoods-and-the-exposure-gate.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ...contracts.pedigree import CandidatePanel, ChromosomeScores, GenomeScores, ParentEligibility
from ...core.errors import InputError

__all__ = ["stack"]

_STACKED = (
    "zero", "two", "pair_explained", "pair_exposed", "junctions", "callable_bins", "transmission_log_likelihood",
)
_EDGES = ("one_floored", "edge_matched", "edge_exposed", "edge_exposed_morgans")


def stack(
    sample_ids: tuple[str, ...],
    panel: CandidatePanel,
    chromosomes: Iterable[ChromosomeScores],
    eligibility: ParentEligibility,
) -> GenomeScores:
    """Own C-contiguous [C,E]/[C,T] arrays; chromosome inputs remain borrowed and unchanged.

    Eligible edges and junction dyads are canonical C layout. The fixed contig
    reductions, rather than this incidental layout, define all sums downstream.
    Every ChromosomeScores.one_floored is consumed verbatim.
    """
    if sample_ids != eligibility.sample_ids:
        raise InputError("score stacking eligibility differs from the sample axis")
    n = len(sample_ids)
    edges = np.ascontiguousarray(np.argwhere(eligibility.parents), dtype=np.int64)
    flat = edges[:, 0] * n + edges[:, 1]
    columns: dict[str, list[np.ndarray]] = {name: [] for name in (*_STACKED, *_EDGES)}
    contigs: list[str]
    contigs, bins, markers, morgans, pair_morgans = [], [], [], [], []
    dyads = np.empty((0, 2), dtype=np.int64)
    for value in chromosomes:
        if value.contig in contigs:
            raise InputError("duplicate informative contig in pedigree scores")
        if not contigs:
            dyads = np.array(value.junction_dyads, dtype=np.int64, order="C", copy=True)
        elif not np.array_equal(value.junction_dyads, dyads):
            raise InputError("chromosome scores orient different dyads")
        if value.zero.shape != (n,) or value.two.shape != (len(panel.trios),):
            raise InputError("chromosome score sample/panel axis differs")
        contigs.append(value.contig)
        bins.append(value.total_bins)
        markers.append(value.informative_markers)
        morgans.append(value.scored_morgans)
        for name in _STACKED:
            columns[name].append(np.array(getattr(value, name), copy=True))
        for name in _EDGES:
            dense = getattr(value, name)
            if dense.shape != (n, n):
                raise InputError(f"chromosome {name} differs from the dense sample axis")
            columns[name].append(np.asarray(dense, dtype=np.float64).reshape(-1)[flat])
        pair_morgans.append(np.array(value.pair_exposed_morgans, dtype=np.float64, copy=True))
        del value, dense
    if not contigs:
        raise InputError("pedigree scores require at least one informative contig")

    def gathered(name: str) -> np.ndarray:
        return np.ascontiguousarray(np.stack(columns.pop(name)), dtype=np.float64)

    return GenomeScores(
        sample_ids,
        tuple(contigs),
        panel,
        edges,
        zero=gathered("zero"),
        one=gathered("one_floored"),
        two=gathered("two"),
        edge_matched=gathered("edge_matched"),
        edge_exposed=gathered("edge_exposed"),
        pair_explained=gathered("pair_explained"),
        pair_exposed=gathered("pair_exposed"),
        total_bins=np.asarray(bins, dtype=np.float64),
        junctions=gathered("junctions"),
        callable_bins=gathered("callable_bins"),
        informative_markers=np.asarray(markers, dtype=np.int64),
        junction_dyads=dyads,
        transmission_log_likelihood=gathered("transmission_log_likelihood"),
        scored_morgans=np.asarray(morgans, dtype=np.float64),
        edge_exposed_morgans=gathered("edge_exposed_morgans"),
        pair_exposed_morgans=np.ascontiguousarray(np.stack(pair_morgans), dtype=np.float64),
    )
