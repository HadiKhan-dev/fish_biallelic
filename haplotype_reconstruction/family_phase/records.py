"""Family-phase product types and checkpoint records.

``identity/family_phase``  per contig: config + identities of the painting, founder-refit and evidence records
                           read, the pedigree edges and the contig's genetic map (bound once)
``family_phase``           per contig: the FamilyPhaseProduct as a dict of arrays and plain data (arrays decode
                           selectively, e.g. ``load_calls``)
run ``family_phase``       the per-contig FamilyPhaseSummary rows
The stage keeps no work state: an interrupted chromosome is recomputed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..evidence import records as evidence_records
from ..founder_refit import records as refit_records
from ..painting import records as painting_records
from ..store.identity import json_digest, stage_identity as build_identity


STAGE = "family_phase"
KEY = "family_phase"
SCHEMA_VERSION = 4
ARRAYS = ("positions", "component_ids", "bin_edges", "bin_edge_offsets", "bin_centers", "calls", "displayed_frame",
          "decoder_frame", "coverage", "coverage_offsets", "frame_undetermined", "phase_breaks", "phase_break_offsets",
          "chain_source", "chain_support", "chain_reset")


@dataclass(frozen=True)
class FamilyPhaseDiagnostics:
    """Per component: sweeps (block sweeps of the default-convention ascent), settled (both ascents' last sweep
    applied no block; False: a sweep cap was reached), objective (the default ascent's log posterior after the
    initialisation and after every sweep, non-decreasing), junction_candidates (distinct candidates of the two
    ascents). Over the chromosome: withheld_joins and tie_breaks (sample phase-set breaks at junction candidates and
    at painting tie boundaries), called_alleles, phased and unphased heterozygous calls, phase sets (with at least
    one phased heterozygous call); elapsed seconds; extra["mirrored_solves"] per component: the mirrored ascent's
    sweeps, settled, objective and junction candidates."""

    sweeps: tuple[int, ...]
    settled: tuple[bool, ...]
    objective: tuple[tuple[float, ...], ...]
    junction_candidates: tuple[int, ...]
    withheld_joins: int
    tie_breaks: int
    called_alleles: int
    phased_heterozygotes: int
    unphased_heterozygotes: int
    phase_sets: int
    elapsed_seconds: float
    extra: dict = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        """Every component's ascent settled before the sweep cap."""
        return all(self.settled)


@dataclass(frozen=True)
class FamilyPhaseProduct:
    """Final phase of one chromosome.

    sample_ids (N, run order); positions int64 [L] (every evidence site of the contig); component_ids int32 [L]
    (assembly component, -1 between components); bin_edges int64 [M] + bin_edge_offsets int64 [C+1] (painting-bin
    edges in bp of each component, half-open bins); bin_centers float64 [M - C] (each bin's centre, the mean
    position of its sites, which the meiosis transitions use; components concatenated); calls int8 [N, L, 2] (released allele per homologue: -1 unknown,
    0 REF, 1 ALT; homologue k of a sample with a released slot-k parent is the slot assigned to that parent: a copy
    of one of its homologues where the chain's source is 0/1, not in not-a-copy stretches); displayed_frame and
    decoder_frame int8 [N, L] (zero: calls are already in the released homologue order; kept for readers of the
    release 2.x layout); coverage int64 [R, 2] + coverage_offsets int64 [N+1] (each sample's painted site ranges
    [start, stop)); edges_identity: identity of the ParentEdges the phase was conditioned on. Released ambiguity:
    frame_undetermined int8 [N, L] (bits 0 and 1 both set on a called heterozygote whose phase is not released);
    phase_breaks int64 [B] + phase_break_offsets int64 [N+1] (per sample, marker boundaries i, between sites i and
    i + 1, where a new phase set starts; component edges are not listed). Meiosis chains (schema 4), one row per
    pedigree edge in ParentEdges order over the chromosome's painting bins (components concatenated in bin_edges
    order): chain_source int8 [E, M - C] (the parent homologue, 0 or 1 in the parent's released order, that the
    child's slot copies with posterior at least phase_posterior in both solves, else -1), chain_support float32 (the
    smaller of the two posteriors there) and chain_reset bool (True at each component's first bin, at meiosis resets
    and at the parent's and the child's released phase-set starts: no inheritance is followed across them).
    """

    contig: str
    sample_ids: tuple[str, ...]
    positions: np.ndarray
    component_ids: np.ndarray
    bin_edges: np.ndarray
    bin_edge_offsets: np.ndarray
    bin_centers: np.ndarray
    calls: np.ndarray
    displayed_frame: np.ndarray
    decoder_frame: np.ndarray
    coverage: np.ndarray
    coverage_offsets: np.ndarray
    edges_identity: str
    diagnostics: FamilyPhaseDiagnostics
    frame_undetermined: np.ndarray | None = None
    phase_breaks: np.ndarray | None = None
    phase_break_offsets: np.ndarray | None = None
    chain_source: np.ndarray | None = None
    chain_support: np.ndarray | None = None
    chain_reset: np.ndarray | None = None

    def component_bin_edges(self, component):
        """Half-open bp bin edges of one component."""
        return self.bin_edges[self.bin_edge_offsets[component]:self.bin_edge_offsets[component + 1]]

    def phase_breaks_of(self, sample):
        """int64 marker boundaries of one sample's phase-set breaks."""
        if self.phase_breaks is None:
            return np.zeros(0, dtype=np.int64)
        return self.phase_breaks[self.phase_break_offsets[sample]:self.phase_break_offsets[sample + 1]]

    def decoder_calls(self):
        """int8 [N, L, 2]: calls in the homologue order recombination decodes (the released order)."""
        gauge = self.displayed_frame ^ self.decoder_frame
        return np.take_along_axis(self.calls, np.stack((gauge, 1 ^ gauge), axis=-1), axis=-1)


@dataclass(frozen=True)
class FamilyPhaseSummary:
    """One row of the family-phase summary table."""

    contig: str
    samples: int
    sites: int
    components: int
    sweeps: int
    settled: bool
    objective: float
    junction_candidates: int
    withheld_joins: int
    tie_breaks: int
    called_alleles: int
    phased_heterozygotes: int
    unphased_heterozygotes: int
    phase_sets: int
    elapsed_seconds: float


def summary(product):
    d = product.diagnostics
    return FamilyPhaseSummary(
        product.contig, len(product.sample_ids), len(product.positions), len(product.bin_edge_offsets) - 1,
        int(sum(d.sweeps)), d.converged, float(sum(o[-1] for o in d.objective if o)),
        int(sum(d.junction_candidates)), d.withheld_joins, d.tie_breaks, d.called_alleles, d.phased_heterozygotes,
        d.unphased_heterozygotes, d.phase_sets, d.elapsed_seconds)


def stage_identity(file, edges, config, chromosome_map) -> dict:
    """Identity of a contig's family phase (module docstring)."""
    return build_identity(STAGE, schema=SCHEMA_VERSION, config=config, upstream={
        "painting": file.info(painting_records.KEY).identity,
        "founder_refit": file.info(refit_records.KEY).identity,
        "evidence": file.info(evidence_records.EVIDENCE).identity,
        "edges": edges.identity,
        "genetic_map": json_digest(chromosome_map.identity()),
    })


def complete(file) -> bool:
    return file.has(KEY)


def progress(file) -> dict:
    """Small status record (no product decoding)."""
    return {"stage": STAGE, "complete": file.has(KEY)}


def encode(product):
    """Record payload: a dict of the product's arrays and plain data."""
    record = {name: getattr(product, name) for name in ARRAYS}
    record.update(schema=SCHEMA_VERSION, contig=product.contig, sample_ids=list(product.sample_ids),
                  edges_identity=product.edges_identity, diagnostics=asdict(product.diagnostics))
    return record


def decode(record):
    record = dict(record)
    if int(record.get("schema", 1)) < SCHEMA_VERSION:
        raise ValueError(f"{record.get('contig')}: family-phase record schema {record.get('schema', 1)} predates "
                         f"the release 3.0 family phase and its meiosis chains; recompute the stage")
    d = dict(record["diagnostics"])
    d.update(sweeps=tuple(d["sweeps"]), settled=tuple(d["settled"]), objective=tuple(tuple(o) for o in d["objective"]),
             junction_candidates=tuple(d["junction_candidates"]))
    diagnostics = FamilyPhaseDiagnostics(**d)
    return FamilyPhaseProduct(
        contig=record["contig"], sample_ids=tuple(record["sample_ids"]),
        **{name: record[name] for name in ARRAYS}, edges_identity=record["edges_identity"],
        diagnostics=diagnostics)


def write(file, product):
    return file.put(KEY, encode(product))


def load(file, *, nthreads=None):
    """The contig's FamilyPhaseProduct."""
    return decode(file.get(KEY, nthreads=nthreads))


def load_calls(file, *, nthreads=None):
    """(sample_ids, positions, component_ids, calls) only: what evaluation and export read."""
    record = file.get(KEY, items=("sample_ids", "positions", "component_ids", "calls"), nthreads=nthreads)
    return tuple(record["sample_ids"]), record["positions"], record["component_ids"], record["calls"]


def load_arrays(file, names, *, nthreads=None):
    """{name: array} of selected product arrays (names among ARRAYS) without decoding the rest."""
    return file.get(KEY, items=tuple(names), nthreads=nthreads)
