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

from .pedigree import PedigreeTable

MANIFEST = "simulation/manifest"
READS = "simulation/reads"
TRUTH = "simulation/truth"
SCHEMA = 1
IN_PROGRESS, COMPLETE = "in_progress", "complete"


@dataclass(frozen=True)
class SimulationManifest:
    """The run-level record of a simulation.

    state: "in_progress" (the realised seed is bound, contigs are being simulated) or "complete".
    requested_seed: the configured seed (None = random); seed: the realised seed. sample_ids: the
    analysed (sequenced) individuals in pedigree order; observed: their pedigree rows (the truth
    evaluation compares with); generated: every generated individual. identity: the source
    identity it was made for.
    """

    state: str
    requested_seed: int | None
    seed: int
    contigs: tuple[str, ...]
    identity: dict
    sample_ids: tuple[str, ...] = ()
    observed: PedigreeTable | None = None
    generated: PedigreeTable | None = None

    def record(self) -> dict:
        return dict(schema=SCHEMA, state=self.state, requested_seed=self.requested_seed, seed=self.seed,
                    contigs=list(self.contigs), identity=self.identity, sample_ids=list(self.sample_ids),
                    observed=None if self.observed is None else self.observed.record(),
                    generated=None if self.generated is None else self.generated.record())

    @classmethod
    def from_record(cls, record) -> "SimulationManifest":
        return cls(state=record["state"], requested_seed=record["requested_seed"], seed=record["seed"],
                   contigs=tuple(record["contigs"]), identity=record["identity"],
                   sample_ids=tuple(record["sample_ids"]),
                   observed=None if record["observed"] is None else PedigreeTable.from_record(record["observed"]),
                   generated=None if record["generated"] is None
                   else PedigreeTable.from_record(record["generated"]))


@dataclass(frozen=True)
class TruthPainting:
    """True founder ancestry of every analysed sample as chunks over bp.

    Sample s owns chunks ``offsets[s]:offsets[s + 1]``; chunk i covers [start[i], end[i]) bp (the
    contig's last chunk also includes its end, the last marker) with founder haplotype hap1[i] on
    the maternal and hap2[i] on the paternal homologue (-1: none). Founder haplotype h is row h of
    ContigTruth.founder_haplotypes.
    """

    offsets: np.ndarray
    start: np.ndarray
    end: np.ndarray
    hap1: np.ndarray
    hap2: np.ndarray


@dataclass(frozen=True)
class TruthCrossovers:
    """One row per gamete of an analysed child: the crossovers its parent's meiosis made.

    child_index: row of the child on the analysed-sample axis; child, parent: names; parent_slot:
    0 = the child's first (maternal) haplotype, 1 = second; cohort_index: 0 = children of the
    founders. Gamete g's crossover positions (bp, float) are positions[offsets[g]:offsets[g + 1]].
    """

    child_index: np.ndarray
    child: tuple[str, ...]
    parent: tuple[str, ...]
    parent_slot: np.ndarray
    cohort_index: np.ndarray
    offsets: np.ndarray
    positions: np.ndarray


@dataclass(frozen=True)
class ContigTruth:
    """The truth of one contig on its full simulated site axis.

    founder_haplotypes: int8[F, L] (founder f = rows 2f, 2f + 1); alleles: int8[N, L, 2] the analysed
    samples' (maternal, paternal) alleles; painting, crossovers: as their types.
    """

    sample_ids: tuple[str, ...]
    positions: np.ndarray
    founder_haplotypes: np.ndarray
    alleles: np.ndarray
    painting: TruthPainting
    crossovers: TruthCrossovers


def truth_painting(ancestry) -> TruthPainting:
    """Chunks of constant (maternal, paternal) founder ancestry from per-haplotype segments.

    ancestry[s] = [maternal segments, paternal segments], each (start, end, founder). Chunk bounds
    are every segment bound of either homologue (bp truncated to integers).
    """
    offsets, starts, ends, hap1s, hap2s = [0], [], [], [], []
    for first, second in ancestry:
        breaks = sorted({value for start, end, _ in (*first, *second) for value in (start, end)})
        first_starts = np.array([start for start, _, _ in first])
        first_ends = np.array([end for _, end, _ in first])
        second_starts = np.array([start for start, _, _ in second])
        second_ends = np.array([end for _, end, _ in second])
        for k in range(len(breaks) - 1):
            start, end = breaks[k], breaks[k + 1]
            if start == end:
                continue
            middle = (start + end) / 2
            owners = []
            for starts_h, ends_h, segments in ((first_starts, first_ends, first),
                                               (second_starts, second_ends, second)):
                owner = -1
                index = np.searchsorted(starts_h, middle, side="right") - 1
                if 0 <= index < len(starts_h) and starts_h[index] <= start and ends_h[index] >= end:
                    owner = segments[index][2]
                owners.append(owner)
            starts.append(int(start))
            ends.append(int(end))
            hap1s.append(owners[0])
            hap2s.append(owners[1])
        offsets.append(len(starts))
    return TruthPainting(offsets=np.asarray(offsets, np.int64), start=np.asarray(starts, np.int64),
                         end=np.asarray(ends, np.int64), hap1=np.asarray(hap1s, np.int64),
                         hap2=np.asarray(hap2s, np.int64))


def truth_crossovers(gametes, child_rows, child_names, cohort_of_row) -> TruthCrossovers:
    """Gametes (child row, slot, parent, positions) of analysed children, on the analysed axis.

    child_rows: {generated row: analysed index}; rows not analysed are left out.
    """
    rows = [(child_rows[row], child_names[row], parent, slot, cohort_of_row[row] - 1, positions)
            for row, slot, parent, positions in gametes if row in child_rows]
    lengths = [len(row[5]) for row in rows]
    return TruthCrossovers(
        child_index=np.asarray([row[0] for row in rows], np.int64),
        child=tuple(row[1] for row in rows), parent=tuple(row[2] for row in rows),
        parent_slot=np.asarray([row[3] for row in rows], np.int8),
        cohort_index=np.asarray([row[4] for row in rows], np.int64),
        offsets=np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
        positions=(np.concatenate([row[5] for row in rows]).astype(np.float64) if rows
                   else np.empty(0, np.float64)))


def save_truth(file, truth: ContigTruth, *, nthreads=None):
    file.put(TRUTH, dict(schema=SCHEMA, sample_ids=truth.sample_ids, positions=truth.positions,
                         founder_haplotypes=truth.founder_haplotypes, alleles=truth.alleles,
                         painting=truth.painting.__dict__, crossovers=truth.crossovers.__dict__),
             nthreads=nthreads)


def load(file, *, nthreads=None) -> ContigTruth:
    """A contig's truth (evaluation only)."""
    payload = file.get(TRUTH, nthreads=nthreads)
    return ContigTruth(sample_ids=tuple(payload["sample_ids"]), positions=payload["positions"],
                       founder_haplotypes=payload["founder_haplotypes"], alleles=payload["alleles"],
                       painting=TruthPainting(**payload["painting"]),
                       crossovers=TruthCrossovers(**payload["crossovers"]))


def save_reads(file, sample_ids, positions, counts, *, nthreads=None):
    file.put(READS, dict(schema=SCHEMA, sample_ids=tuple(sample_ids), positions=np.asarray(positions, np.int64),
                         counts=np.ascontiguousarray(counts, dtype=np.int32)), nthreads=nthreads)


def load_reads(file, *, nthreads=None) -> dict:
    return file.get(READS, nthreads=nthreads)


def load_manifest(store) -> SimulationManifest | None:
    return SimulationManifest.from_record(store.run.get(MANIFEST)) if store.run.has(MANIFEST) else None


def truth_pedigree_table(manifest: SimulationManifest) -> tuple[list[str], list[list[str]]]:
    """Rows of OUTPUT/simulation/truth_pedigree.csv: the analysed samples and their true parents."""
    table = manifest.observed
    header = ["sample", "cohort", "parent1", "parent2"]
    return header, [[sample, f"F{cohort}", first, second] for sample, cohort, first, second in
                    zip(table.samples, table.cohorts, table.parent1, table.parent2)]
