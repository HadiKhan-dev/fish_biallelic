"""Family-phase product types and checkpoint records.

``identity/family_phase``  per contig: config + identities of the painting, founder-refit and evidence records
                           read, the pedigree edges and the contig's genetic map (bound once)
``family_phase``           per contig: the FamilyPhaseProduct as a dict of arrays and plain data (arrays decode
                           selectively, e.g. ``load_calls``)
run ``family_phase``       the per-contig FamilyPhaseSummary rows
(Release 2.1 keeps no work state: the family MAP solver is cheap and an interrupted chromosome is recomputed.)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..evidence import records as evidence_records
from ..founder_refit import records as refit_records
from ..painting import records as painting_records
from ..store.identity import json_digest, stage_identity as build_identity
from .frames import ChildFrameChange, FrameDecision, FrameRepair, GuardBlock


STAGE = "family_phase"
KEY = "family_phase"
SCHEMA_VERSION = 2
ARRAYS = ("positions", "component_ids", "bin_edges", "bin_edge_offsets", "calls", "displayed_frame",
          "decoder_frame", "coverage", "coverage_offsets", "frame_undetermined", "phase_breaks", "phase_break_offsets")
# Arrays added in release 2.2 (absent from schema-1 payloads; decoded as 'nothing undetermined').
ARRAYS_2_2 = ("frame_undetermined", "phase_breaks", "phase_break_offsets")


@dataclass(frozen=True)
class FamilyPhaseDiagnostics:
    """map_sweeps: family-MAP sweeps run (every block once, then the relabelling scan, the compound moves and the
    site moves; the last accepted nothing); converged: the solver stopped because a sweep accepted nothing (always
    true for a released product: reaching map_max_sweeps is an error); map_accepted: accepted steps per sweep
    (block intervals, relabelling, compound and site moves);
    map_objective: the family model's log posterior J after the initialisation and after every sweep (nats,
    non-decreasing); polish_sweep_updates: accepted vertex updates per polish sweep; called_alleles: called entries
    of ``calls``; changed_observable_phase_sites: two-parent sample-sites at heterozygous scaffold sites where the
    polished frame differs from the family MAP frame; frame_repair: root-frame decisions (masks dropped).
    map_conflicts: Mendelian-conflict diagnostic, (parent sample, bp position) of every site where the parent's best
    genotype leaves a child's allele mistransmitted in the MAP (a root's best summed genotype).
    map_two_schedules (release 2.2): the two-schedule estimator's record (J of the canonical and mirrored end points,
    which was the base, clusters, transplanted and tied transplants, released J); map_sweeps / map_accepted /
    map_objective describe the final ascent.
    (Release 2.1; 2.0 recorded belief-propagation ``iterations`` and ``residuals`` instead of the map_* fields:
    such payloads decode with map_sweeps = iterations and empty map_accepted / map_objective / map_conflicts.)"""

    map_sweeps: int
    converged: bool
    map_accepted: tuple[int, ...]
    map_objective: tuple[float, ...]
    polish_sweep_updates: tuple[int, ...]
    called_alleles: int
    changed_observable_phase_sites: int
    frame_repair: FrameRepair
    elapsed_seconds: float
    map_conflicts: tuple = ()
    map_two_schedules: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FamilyPhaseProduct:
    """Final phase of one chromosome.

    sample_ids (N, run order); positions int64 [L] (every evidence site of the contig); component_ids int32 [L]
    (assembly component, -1 between components); bin_edges int64 [M] + bin_edge_offsets int64 [C+1] (painting-bin
    edges in bp of each component, half-open bins; phase is rigid inside a bin); calls int8 [N, L, 2] (allele per
    track: -1 unknown, 0 REF, 1 ALT; displayed track order); displayed_frame int8 [N, L] and decoder_frame int8
    [N, L] (1 = the painting scaffold's tracks swapped; calls[n, j, k] = scaffold[n, j, k ^ displayed_frame], the
    family model's internal frame is decoder_frame, so recombination reads calls[..., k ^ displayed ^ decoder]);
    coverage int64 [R, 2] + coverage_offsets int64 [N+1] (each sample's painted site ranges [start, stop));
    edges_identity: identity of the ParentEdges the phase was conditioned on. Released ambiguity (release 2.2,
    stage.released_ambiguity): frame_undetermined int8 [N, L] (bit 0: decoder frame, bit 1: displayed frame
    undetermined there; a displayed-undetermined heterozygous or half-called site has no determined phase);
    phase_breaks int64 [B] + phase_break_offsets int64 [N+1] (per sample, marker boundaries i, between sites i and
    i + 1, where the displayed phase of the right side relative to the left side is not determined by data; component
    edges are not listed).
    """

    contig: str
    sample_ids: tuple[str, ...]
    positions: np.ndarray
    component_ids: np.ndarray
    bin_edges: np.ndarray
    bin_edge_offsets: np.ndarray
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

    def component_bin_edges(self, component):
        """Half-open bp bin edges of one component."""
        return self.bin_edges[self.bin_edge_offsets[component]:self.bin_edge_offsets[component + 1]]

    def phase_breaks_of(self, sample):
        """int64 marker boundaries of one sample's phase-set breaks (released ambiguity)."""
        if self.phase_breaks is None:
            return np.zeros(0, dtype=np.int64)
        return self.phase_breaks[self.phase_break_offsets[sample]:self.phase_break_offsets[sample + 1]]

    def decoder_calls(self):
        """int8 [N, L, 2]: calls in the decoder (internal family) frame."""
        gauge = self.displayed_frame ^ self.decoder_frame
        return np.take_along_axis(self.calls, np.stack((gauge, 1 ^ gauge), axis=-1), axis=-1)


@dataclass(frozen=True)
class FamilyPhaseSummary:
    """One row of the family-phase summary table."""

    contig: str
    samples: int
    sites: int
    map_sweeps: int
    converged: bool
    map_objective: float
    called_alleles: int
    changed_observable_phase_sites: int
    root_frame_moves: int
    root_frame_reverts: int
    root_frame_transfers: int
    elapsed_seconds: float


def summary(product):
    diagnostics = product.diagnostics
    repair = diagnostics.frame_repair
    return FamilyPhaseSummary(
        product.contig, len(product.sample_ids), len(product.positions), diagnostics.map_sweeps,
        diagnostics.converged, float(diagnostics.map_objective[-1]) if diagnostics.map_objective else float("nan"),
        diagnostics.called_alleles, diagnostics.changed_observable_phase_sites, len(repair.moves),
        len(repair.reverts), len(repair.transfers), diagnostics.elapsed_seconds)


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
    if record["frame_undetermined"] is None:          # a product built without released ambiguity
        samples, sites = np.asarray(product.calls).shape[:2]
        record.update(frame_undetermined=np.zeros((samples, sites), dtype=np.int8),
                      phase_breaks=np.zeros(0, dtype=np.int64),
                      phase_break_offsets=np.zeros(samples + 1, dtype=np.int64))
    record.update(schema=SCHEMA_VERSION, contig=product.contig, sample_ids=list(product.sample_ids),
                  edges_identity=product.edges_identity, diagnostics=asdict(product.diagnostics))
    return record


def _decode_diagnostics(record):
    repair = record["frame_repair"]
    frame_repair = FrameRepair(
        moves=tuple(FrameDecision(**item) for item in repair["moves"]),
        reverts=tuple(FrameDecision(**item) for item in repair["reverts"]),
        transfers=tuple(FrameDecision(**item) for item in repair["transfers"]),
        child_frames=tuple(ChildFrameChange(**item) for item in repair["child_frames"]),
        rejected=tuple(tuple(item) for item in repair["rejected"]),
        guard_blocks=tuple(GuardBlock(**item) for item in repair["guard_blocks"]),
        flagged_edges=tuple(tuple(item) for item in repair["flagged_edges"]))
    record = dict(record)
    if "iterations" in record:                   # a 2.0 belief-propagation payload
        record["map_sweeps"] = record.pop("iterations")
        record.pop("residuals", None)
        record.setdefault("map_accepted", ())
        record.setdefault("map_objective", ())
    record["map_conflicts"] = tuple(tuple(item) for item in record.get("map_conflicts", ()))
    return FamilyPhaseDiagnostics(**{**record, "frame_repair": frame_repair})


def decode(record):
    record = dict(record)
    if "frame_undetermined" not in record:          # a schema-1 (release 2.0/2.1) payload
        samples, sites = np.asarray(record["calls"]).shape[:2]
        record.update(frame_undetermined=np.zeros((samples, sites), dtype=np.int8),
                      phase_breaks=np.zeros(0, dtype=np.int64),
                      phase_break_offsets=np.zeros(samples + 1, dtype=np.int64))
    return FamilyPhaseProduct(
        contig=record["contig"], sample_ids=tuple(record["sample_ids"]),
        **{name: record[name] for name in ARRAYS}, edges_identity=record["edges_identity"],
        diagnostics=_decode_diagnostics(record["diagnostics"]))


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
