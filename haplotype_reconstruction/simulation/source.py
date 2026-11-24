"""Generate a simulation's typed contig inputs without storage or scheduling effects.

Owns founder materialisation, observation selection and the reads/truth transition
(OU23, OU29, OU30; docs/methods/simulation.md). The stage binds the realised seed
and supplies templates, maps and explicit pedigree rows. It publishes the results.

Draw order: founders use SeedSequence([seed, 2000000]).spawn(C)[c]; pedigree
uses the master stream (one initial Python structure seed for random cohorts only),
then a contig-major child-seed block per cohort/depth. Observation uses its separate
SeedSequence stream. Read seeds are drawn for EVERY contig before any resume skip;
read workers consume sample blocks with independently indexed per-sample streams.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..contracts.genetic_map import ChromosomeGeneticMap
from ..contracts.roles import freeze_record_arrays, record_field
from ..contracts.source import ContigReads
from ..core.errors import InputError, SettingsError
from ..core.random import read_seeds, stream
from .artefacts import ArtefactModel, ArtefactResult, apply_artefacts
from .design import PedigreeRow, ReadModel, SimulationConfig, observed_indices
from .genealogy import mutations, separating_length
from .haplotypes import Extent, Mosaic, alleles, founder_mosaic, source_at
from .pedigree import PedigreeTable, materialise_founders, pair_founders, simulate_explicit_pedigree, simulate_pedigree
from .reads import (
    ReadInputs,
    contamination_levels,
    duplicate_rates,
    exogenous_shares,
    library_error_factors,
    read_inputs,
)
from .sites import draw_site_set, realise_profile, relative_morgans
from .stocks import stock_founders, stock_record, stock_stream
from .structure import (
    CollapsedParalogs,
    ContigStructure,
    StructureReads,
    StructureTruth,
    collapsed_copies,
    collapsed_psv_sites,
    copy_alleles,
    draw_collapsed,
    draw_structure,
    haplotype_carriers,
    paralog_sites,
)
from .truth import ContigTruth, TruthCrossovers, truth_crossovers, truth_flips, truth_founder_mosaics, truth_painting

if TYPE_CHECKING:
    from numpy.typing import NDArray

    # ContigReads.__post_init__ accepts these int64 draws and freezes them as int32.
    _ContigReadsFactory = Callable[[str, tuple[str, ...], NDArray[np.int64], NDArray[np.int64]], ContigReads]

__all__ = [
    "FounderTemplate",
    "SourcePanel",
    "ContigSimulation",
    "SimulationPlan",
    "prepare_simulation",
    "contig_read_inputs",
    "contig_result",
]


@dataclass(frozen=True)
class _GeneratingMap:
    """Use the specified cumulative-coordinate difference for every interval.

    The no-map convenience method multiplies bp span by rate, which can differ by
    an ulp from g(right)-g(left). R5 and the site axis require the latter exactly.
    """

    chromosome: ChromosomeGeneticMap = record_field("obs", dtype="ChromosomeGeneticMap")

    def cumulative_morgans(self, positions):
        return self.chromosome.cumulative_morgans(positions)

    def interval_morgans(self, left, right):
        return self.cumulative_morgans(right) - self.cumulative_morgans(left)


@dataclass(frozen=True)
class _CrossSites:
    """Marker coordinates plus an unobserved terminal marker and the extent start.

    The released meiosis reads scalar zero to start its physical clock, and uses
    NumPy search/index operations for marker mutations. Keep those coordinates
    separate: a zero mutation gap at the extent start maps to the first real site.
    Only the terminal sentinel can be flipped outside the retained marker axis.
    """

    positions: np.ndarray = record_field("obs", axes=("marker",), units="bp", dtype="int64")
    start_bp: float = record_field("obs", units="bp", dtype="float64")

    def __post_init__(self):
        freeze_record_arrays(self)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)) and key == 0:
            return self.start_bp
        return self.positions[key]

    def __array__(self, dtype=None, copy=None):
        if copy:
            return np.array(self.positions, dtype=dtype, copy=True)
        return np.asarray(self.positions, dtype=dtype)


@dataclass(frozen=True)
class SourcePanel:
    """Compact source rows in REF/ALT coding with stable IDs independent of rows."""

    positions: np.ndarray = record_field("obs", axes=("L",), units="bp", dtype="int64")
    alleles: np.ndarray = record_field("obs", axes=("S", "L"), units="1", dtype="int8")
    source_id: np.ndarray = record_field("obs", axes=("S",), units="index", dtype="int32")
    population: np.ndarray = record_field("obs", axes=("S",), units="index", dtype="int32")
    reference_derived: np.ndarray = record_field("obs", axes=("L",), units="1", dtype="bool")
    total_sources: int = record_field("obs", units="1", dtype="int")
    spurious: np.ndarray | None = record_field("obs", axes=("L",), units="1", dtype="bool | None", default=None)

    def __post_init__(self):
        for name, dtype in (
            ("positions", np.int64),
            ("alleles", np.int8),
            ("source_id", np.int32),
            ("population", np.int32),
            ("reference_derived", bool),
        ):
            value = np.asarray(getattr(self, name))
            if name == "alleles" and np.any((value != 0) & (value != 1)):
                raise InputError("source alleles must be 0 or 1")
            object.__setattr__(self, name, np.asarray(value, dtype=dtype))
        if self.positions.ndim != 1 or np.any(np.diff(self.positions) <= 0):
            raise InputError("source positions must be increasing")
        if self.source_id.ndim != 1:
            raise InputError("source IDs must have one row axis")
        n = len(self.source_id)
        if (
            self.alleles.shape != (n, len(self.positions))
            or self.population.shape != (n,)
            or self.reference_derived.shape != self.positions.shape
            or np.any(np.diff(self.source_id) <= 0)
            or np.any(self.source_id < 0)
            or np.any(self.source_id >= self.total_sources)
            or np.any(self.population < 0)
        ):
            raise InputError("source panel axes or stable source IDs are invalid")
        mark = np.zeros(len(self.positions), bool) if self.spurious is None else np.asarray(self.spurious, bool)
        if mark.shape != self.positions.shape or np.any(self.alleles[:, mark]) or np.any(self.reference_derived[mark]):
            raise InputError("spurious source sites must be reference-only and aligned with positions")
        object.__setattr__(self, "spurious", mark)
        freeze_record_arrays(self)


@dataclass(frozen=True)
class FounderTemplate:
    """Validated file input: fixed alleles or legacy probabilities, before any draw.

    Fixed unknown alleles (-1) require unknown="draw". Legacy ties use the released
    per-contig stream in haplotype order. Population IDs follow population_names.
    """

    positions: np.ndarray = record_field("obs", axes=("L",), units="bp", dtype="int64")
    allele_probabilities: np.ndarray | None = record_field(
        "obs", axes=("H", "L", "2"), units="probability", dtype="float64 | None", default=None
    )
    alleles: np.ndarray | None = record_field("obs", axes=("H", "L"), units="1", dtype="int8 | None", default=None)
    unknown: str = record_field("obs", units="1", dtype="str", default="refuse")
    population: np.ndarray | None = record_field("obs", axes=("H",), units="index", dtype="int32 | None", default=None)
    population_names: tuple[str, ...] = record_field("obs", units="1", dtype="tuple[str, ...]", default=("",))

    def __post_init__(self) -> None:
        raw_positions = np.asarray(self.positions)
        if raw_positions.dtype.kind not in "iu":
            raise InputError("file source positions must be integer bp values")
        if np.any(raw_positions > np.iinfo(np.int64).max):
            raise InputError("file source positions must fit int64 bp coordinates")
        positions = np.asarray(raw_positions, dtype=np.int64)
        if positions.ndim != 1 or len(positions) == 0 or np.any(np.diff(positions) <= 0):
            raise InputError("founder template positions must be non-empty and increasing")
        if (self.alleles is None) == (self.allele_probabilities is None):
            raise InputError("file source requires exactly one of alleles or allele_probabilities")
        if self.unknown not in ("refuse", "draw"):
            raise InputError("file source unknown must be refuse or draw")
        if self.alleles is not None:
            values = np.asarray(self.alleles)
            if values.ndim != 2 or values.shape[1:] != (len(positions),):
                raise InputError("source allele axes must be (haplotypes, sites)")
            if np.any(~np.isin(values, (-1, 0, 1))) or (self.unknown != "draw" and np.any(values == -1)):
                raise InputError("source alleles must be 0 or 1; -1 requires unknown=draw")
            object.__setattr__(self, "alleles", values.astype(np.int8))
        else:
            values = np.asarray(self.allele_probabilities)
            if values.ndim != 3 or values.shape[1:] != (len(positions), 2):
                raise InputError("founder template axes must be (haplotypes, sites, 2)")
            if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values.sum(axis=2) <= 0):
                raise InputError("founder template probabilities must be finite, nonnegative, with positive mass")
            object.__setattr__(self, "allele_probabilities", values)
        if len(values) == 0 or len(values) % 2:
            raise InputError("founder templates must hold a positive even number of haplotypes")
        population = (
            np.zeros(len(values), np.int32) if self.population is None else np.asarray(self.population, np.int32)
        )
        if (
            population.shape != (len(values),)
            or np.any(population < 0)
            or np.any(population >= len(self.population_names))
        ):
            raise InputError("source population axis is invalid")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "population", population)
        freeze_record_arrays(self)

    def materialise(self, rng) -> SourcePanel:
        """Resolve a file using only its released per-contig founder stream."""
        if self.alleles is None:
            values = np.asarray(materialise_founders(self.allele_probabilities, rng), np.int8)
        else:
            values = self.alleles.copy()
            for row in values:
                unknown = row == -1
                if np.any(unknown):
                    row[unknown] = rng.integers(0, 2, size=int(unknown.sum()), dtype=np.int8)
        return SourcePanel(
            self.positions,
            values,
            np.arange(len(values), dtype=np.int32),
            self.population,
            np.zeros(len(self.positions), bool),
            len(values),
        )


@dataclass(frozen=True)
class ContigSimulation:
    """A whole-contig reads task with its already drawn seed and observed truth.

    Offspring and ancestry hold [N][2] Mosaic pairs; no allele arrays are retained.
    ``contig_read_inputs`` prepares read factors; sample tasks generate counts and
    ``contig_result`` combines counts with unchanged truth. These operations do not mutate
    this record. Each sample has its own stream (simulation/reads.py), so the sample
    axis may be split into independently simulated blocks.
    """

    contig: str = record_field("obs", axes=(), units="1", dtype="str")
    sample_ids: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    positions: np.ndarray = record_field("obs", axes=(), units="1", dtype="np.ndarray")
    offspring: list = record_field("obs", axes=(), units="1", dtype="list")
    ancestry: list = record_field("obs", axes=(), units="1", dtype="list")
    founders: list = record_field("obs", axes=(), units="1", dtype="list")
    panel: SourcePanel = record_field("obs", units="1", dtype="SourcePanel")
    crossovers: TruthCrossovers = record_field("obs", axes=(), units="1", dtype="TruthCrossovers")
    depth: float = record_field("obs", axes=(), units="1", dtype="float")
    read_model: ReadModel = record_field("obs", axes=(), units="1", dtype="ReadModel")
    seed: int = record_field("obs", axes=(), units="1", dtype="int")
    generating_seed: int = record_field("obs", dtype="int", default=0)
    artefacts: ArtefactModel = record_field("obs", dtype="ArtefactModel", default_factory=ArtefactModel)
    dropped_positions: np.ndarray = record_field(
        "obs", axes=("dropped",), units="bp", dtype="int64", default_factory=lambda: np.empty(0, np.int64)
    )
    structure: ContigStructure | None = record_field("obs", dtype="ContigStructure | None", default=None)
    # With structure: (offsets int64[G + 1], positions float64) of reference-axis switches, gametes as in crossovers.
    switches: tuple | None = record_field("obs", dtype="tuple | None", default=None)
    # Collapsed paralogs (simulation/structure.py), or None when off; the panel already holds their PSV sites.
    collapsed: CollapsedParalogs | None = record_field("obs", dtype="CollapsedParalogs | None", default=None)
    # Unsequenced contamination donors (explicit pedigrees): identifiers and their haplotype pairs (Mosaic).
    donor_ids: tuple[str, ...] = record_field("obs", dtype="tuple[str, ...]", default=())
    donor_haplotypes: tuple = record_field("obs", dtype="tuple", default=())

    def __post_init__(self):
        freeze_record_arrays(self)


@dataclass(frozen=True)
class SimulationPlan:
    """Shared pedigree truth and contig tasks in the configured contig order."""

    generated: PedigreeTable = record_field("obs", axes=(), units="1", dtype="PedigreeTable")
    observed: PedigreeTable = record_field("obs", axes=(), units="1", dtype="PedigreeTable")
    contigs: tuple[ContigSimulation, ...] = record_field(
        "obs", axes=(), units="1", dtype="tuple[ContigSimulation, ...]"
    )


def _materialise_contig_panel(task) -> SourcePanel:
    """Module-scope callback for ordered contig materialisation, including thread pools."""
    template, seed, index = task
    return template.materialise(stream("simulation.founders", seed=seed, index=index))


def prepare_simulation(
    config: SimulationConfig,
    seed: int,
    templates: Sequence[FounderTemplate],
    chromosome_maps: Sequence[ChromosomeGeneticMap],
    *,
    pedigree_rows: tuple[PedigreeRow, ...] | None = None,
    cores: int = 1,
    runner: Callable | None = None,
    map_work: Callable | None = None,
) -> SimulationPlan:
    """Materialise founders and shared pedigree; return tasks with all read seeds bound.

    Templates/maps follow config.contigs order. A supplied seed is already realised.
    Explicit rows override cohorts and retain table observation. Input axes are not
    mutated; unequal founder counts across contigs raise InputError before draws.
    The runner executes independent child batches and returns indexed results.
    ``map_work(function, items)`` (ordered, optional) draws the contigs' founders concurrently, each
    from its own stream, so the founders are the same.
    """
    if len(templates) != len(config.contigs) or len(chromosome_maps) != len(config.contigs):
        raise InputError("templates and generating maps must follow the configured contig axis")
    if config.structure.enabled:
        raise InputError("structural variants are generated contig by contig (prepare_contig), not by this planner")
    if len({len(t.population) for t in templates}) != 1:
        raise InputError("simulation contigs have different founder counts")
    sites = [t.positions for t in templates]

    panel_tasks = [(template, seed, c) for c, template in enumerate(templates)]
    panels = list(
        map(_materialise_contig_panel, panel_tasks)
        if map_work is None
        else map_work(_materialise_contig_panel, panel_tasks)
    )
    founders = [
        pair_founders([founder_mosaic(panel.positions[0], int(h), int(h)) for h in panel.source_id]) for panel in panels
    ]
    common: dict[str, Any] = dict(
        seed=seed,
        recombination_rate=chromosome_maps[0].fallback_rate_per_bp,
        mutation_rate=config.mutation_rate_per_bp,
        chromosome_maps=[m if m.has_map else None for m in chromosome_maps],
        budget=cores,
        runner=runner,
    )
    if pedigree_rows is not None:
        generated, simulated = simulate_explicit_pedigree(founders, sites, pedigree_rows, **common)
        selected = np.asarray([i for i, row in enumerate(pedigree_rows) if row.observed], dtype=np.int64)
    else:
        generated, simulated = simulate_pedigree(founders, sites, config.cohorts, design=config.design, **common)
        selected = observed_indices(generated.cohorts, config.design, seed)
    observed = generated.take(selected)
    analysed = {int(row): index for index, row in enumerate(selected)}
    seeds = read_seeds(seed, len(config.contigs))
    tasks = tuple(
        ContigSimulation(
            contig,
            observed.samples,
            sites[c],
            [simulated[c].haplotypes[row] for row in selected],
            [simulated[c].ancestry[row] for row in selected],
            founders[c],
            panels[c],
            truth_crossovers(simulated[c].gametes, analysed, generated.samples, generated.cohorts),
            config.depth,
            config.reads,
            seeds[c],
            seed,
            config.artefacts,
        )
        for c, contig in enumerate(config.contigs)
    )
    return SimulationPlan(generated, observed, tasks)


def contig_read_inputs(task: ContigSimulation) -> ReadInputs:
    """The contig's haplotypes, shared read factors and per-sample stream seed (from its task-local stream)."""
    rng = stream("simulation.contig_reads", seed=task.seed)

    # alleles() uses a source-ID-indexed panel; compact rows are remapped without
    # allocating a total_sources by sites table.
    def gather(mosaic):
        if not len(task.positions):
            return np.empty(0, np.int8)
        ids = source_at(mosaic, task.positions)
        rows = np.searchsorted(task.panel.source_id, ids)
        if np.any(rows >= len(task.panel.source_id)) or np.any(task.panel.source_id[rows] != ids):
            raise InputError("mosaic names a source absent from the panel at a retained marker")
        # Unretained segments need no compact row. Their arbitrary row zero is
        # never gathered; all source IDs at retained markers were checked above.
        segment_rows = np.searchsorted(task.panel.source_id, mosaic.source)
        segment_rows = np.minimum(segment_rows, len(task.panel.source_id) - 1)
        compact = Mosaic(mosaic.starts, segment_rows, mosaic.founder, mosaic.flips)
        return alleles(compact, task.panel.alleles, task.positions)

    hap0 = np.array([gather(individual[0]) for individual in task.offspring], dtype=np.int8)
    hap1 = np.array([gather(individual[1]) for individual in task.offspring], dtype=np.int8)
    donor0 = donor1 = None
    if task.donor_ids:
        donor0 = np.array([gather(individual[0]) for individual in task.donor_haplotypes], dtype=np.int8)
        donor1 = np.array([gather(individual[1]) for individual in task.donor_haplotypes], dtype=np.int8)
    injected = None
    if task.read_model.injected_outlier_rate > 0:
        # G1i: ordered genotypes replaced before any read is drawn (simulation/injection.py, its own streams)
        from .injection import inject

        original = (hap0, hap1)
        hap0, hap1, injected = inject(hap0, hap1, task.read_model.injected_outlier_rate,
                                      task.read_model.injected_outlier_pattern, task.seed, tuple(task.sample_ids))
    inputs = read_inputs(hap0, hap1, task.positions, task.depth, task.read_model, rng)
    if injected is not None:
        inputs = replace(inputs, original0=original[0], original1=original[1], injected=injected)
    if task.read_model.exogenous_read_share:
        # Each sample's share is genome-wide: keyed by the generating seed and its identifier, not by contig.
        shares = exogenous_shares(task.sample_ids, task.read_model, task.generating_seed)
        inputs = replace(inputs, exogenous_share=shares)
    if task.read_model.contamination_share or task.read_model.contamination_samples:
        # Each library's share and donor are genome-wide: keyed by the generating seed and its identifier.
        shares, donors = contamination_levels(task.sample_ids, task.read_model, task.generating_seed,
                                              extra_donors=task.donor_ids)
        inputs = replace(inputs, contamination_share=shares, contamination_donor=donors, donor_hap0=donor0,
                         donor_hap1=donor1)
    if task.read_model.duplicate_rate:
        # Each library's duplicate rate is genome-wide: keyed by the generating seed and its identifier.
        inputs = replace(inputs, duplicate_rate=duplicate_rates(task.sample_ids, task.read_model, task.generating_seed))
    if task.read_model.library_error_sigma:
        # Each library's error factor is genome-wide: keyed by the generating seed and its identifier.
        inputs = replace(inputs, error_factor=library_error_factors(task.sample_ids, task.read_model,
                                                                    task.generating_seed))
    if task.collapsed is not None:
        # The copy's base: ALT at a PSV, the ancestral base elsewhere inside a segment (REF unless the reference genome
        # holds the derived allele), -1 outside every segment.
        inside = task.collapsed.index(task.positions) >= 0
        psv = collapsed_psv_sites(task.collapsed, task.positions)
        allele = np.where(inside, (psv | task.panel.reference_derived).astype(np.int8), np.int8(-1)).astype(np.int8)
        inputs = replace(
            inputs,
            collapsed_start=task.collapsed.start,
            collapsed_end=task.collapsed.end,
            collapsed_copies=collapsed_copies(task.offspring, task.collapsed),
            collapsed_allele=np.ascontiguousarray(allele),
            collapsed_donor_copies=(
                collapsed_copies(list(task.donor_haplotypes), task.collapsed) if task.donor_ids else None
            ),
        )
    if task.structure is None:
        return inputs
    paralog = paralog_sites(task.structure, task.positions)
    present, copies = copy_alleles(
        task.offspring, task.structure, task.positions, task.panel.alleles, task.panel.source_id, paralog
    )
    return replace(inputs, structure=StructureReads(present, copies, *paralog))


def contig_result(
    task: ContigSimulation, inputs: ReadInputs, counts: NDArray[np.integer]
) -> tuple[ContigReads, ContigTruth]:
    """Reads (counts int[N, L, 2] in the observed row order, frozen as int32) and unchanged truth of one contig."""
    if task.artefacts.enabled:
        transformed = apply_artefacts(
            inputs.hap0 + inputs.hap1,
            np.asarray(counts, np.int32),
            task.positions,
            task.generating_seed,
            task.contig,
            task.sample_ids,
            task.artefacts,
            heterozygote_alt_probability=task.read_model.heterozygote_alt_probability,
            library_factor=inputs.error_factor,
        )
    else:
        transformed = ArtefactResult(counts, None, None)
    truth = ContigTruth(
        sample_ids=task.sample_ids,
        positions=task.positions,
        # the original alleles (ancestry); with G1i's injection the replaced ones and the mask beside them
        alleles=np.ascontiguousarray(np.stack(
            (inputs.hap0, inputs.hap1) if inputs.injected is None else (inputs.original0, inputs.original1), axis=2
        ), dtype=np.int8),
        injected_alleles=None if inputs.injected is None else np.ascontiguousarray(
            np.stack((inputs.hap0, inputs.hap1), axis=2), dtype=np.int8),
        injected_mask=inputs.injected,
        painting=truth_painting(task.offspring, task.positions),
        crossovers=task.crossovers,
        source_haplotypes=task.panel.alleles,
        source_id=task.panel.source_id,
        source_population=task.panel.population,
        total_sources=task.panel.total_sources,
        reference_derived=task.panel.reference_derived,
        founder_mosaics=truth_founder_mosaics(task.founders, task.positions),
        source_painting=truth_painting(task.offspring, task.positions, source=True),
        flips=truth_flips(task.offspring),
        dropped_positions=task.dropped_positions,
        missed=transformed.missed,
        systematic_site=transformed.systematic_site,
        spurious=task.panel.spurious,
        structure=None if task.structure is None else _structure_truth(task, inputs),
        **_library_truth(task, inputs),
    )
    return cast("_ContigReadsFactory", ContigReads)(
        task.contig, task.sample_ids, task.positions, transformed.counts
    ), truth


def _library_truth(task: ContigSimulation, inputs: ReadInputs) -> dict:
    """Realised per-library contamination (share and donor identifier, "" when clean) and PCR duplicate rates, for the
    truth; empty when both modes are off (the released record)."""
    out = {}
    if inputs.contamination_share is not None:
        names = (*map(str, task.sample_ids), *task.donor_ids)
        out["contamination_share"] = np.asarray(inputs.contamination_share, np.float64)
        out["contamination_donor"] = tuple("" if d < 0 else names[d] for d in inputs.contamination_donor)
    if inputs.duplicate_rate is not None:
        out["duplicate_rate"] = np.asarray(inputs.duplicate_rate, np.float64)
    if inputs.error_factor is not None:
        out["library_error_factor"] = np.asarray(inputs.error_factor, np.float64)
    return out


def _structure_truth(task: ContigSimulation, inputs: ReadInputs) -> StructureTruth:
    """The contig's structure on the analysed axes (StructureTruth)."""
    structure = task.structure
    assert structure is not None and task.switches is not None and inputs.structure is not None
    founders = haplotype_carriers(task.founders, structure).reshape(2 * len(task.founders), len(structure))
    return StructureTruth(
        kind=structure.kind,
        start=structure.start,
        end=structure.end,
        target=structure.target,
        founder_carries=founders,
        sample_carries=haplotype_carriers(task.offspring, structure),
        switch_offsets=task.switches[0],
        switch_positions=task.switches[1],
        paralog_offsets=inputs.structure.paralog_offsets,
        paralog_sites=inputs.structure.paralog_sites,
        **({} if inputs.collapsed_copies is None else dict(
            collapsed_start=inputs.collapsed_start,
            collapsed_end=inputs.collapsed_end,
            collapsed_copies=inputs.collapsed_copies,
            collapsed_psv_sites=np.flatnonzero(collapsed_psv_sites(task.collapsed, task.positions)).astype(np.int64),
        )),
    )


def prepare_stocks(config, seed, rows):
    """Stable source layout and shared pedigrees; all capacities have been validated."""
    order = config.founder_stocks(rows)
    records, first = [], 0
    for specification in config.stocks:
        records.append(stock_record(specification.name, specification, order.count(specification.name), first, seed))
        first += 2 * specification.founders
    return tuple(records), order


def cross_founders(config, records, order, extent, chromosome_map, seed, contig, structure=None):
    """Take whole individuals in pedigree order, then assign the cross-founder axis."""
    taken = {r.name: iter(stock_founders(r, extent, chromosome_map, seed, contig, structure)) for r in records}
    result = []
    for i, name in enumerate(order):
        pair = next(taken[name])
        result.append(
            tuple(
                Mosaic(h.starts, h.source, np.full(len(h.starts), 2 * i + slot, np.int32), h.flips)
                for slot, h in enumerate(pair)
            )
        )
    return result


def carried_sets(founders, extent, chromosome_map, total_sources):
    """Right-continuous carried sets in relative Morgans, reference never carried."""
    length = float(relative_morgans(extent, chromosome_map, np.array([extent.end_bp]))[0])
    haplotypes = [h for pair in founders for h in pair]
    mapped = [relative_morgans(extent, chromosome_map, h.starts) for h in haplotypes]
    boundaries = np.unique(np.concatenate([np.array([0.0, length]), *mapped]))
    if length == 0:
        boundaries = np.array([0.0, 0.0])
    carried = np.zeros((len(boundaries) - 1, total_sources + 1), bool)
    for h, x in zip(haplotypes, mapped, strict=True):
        ids = h.source[np.searchsorted(x, boundaries[:-1], side="right") - 1]
        carried[np.arange(len(carried)), ids] = True
    return boundaries, carried


def coalescent_panel(config, records, founders, extent, chromosome_map, seed, contig):
    """Two genealogy passes, independent profile/site clocks, REF and compact rows."""
    names, parent, split, size = config.source.population_tree()
    populations = np.concatenate([np.full(2 * int(r.sizes[0]), names.index(r.population), np.int64) for r in records])
    total = len(populations)
    reference = names.index(config.source.reference_population) if config.source.reference_population else 0
    leaves = np.r_[populations, np.int64(reference)]
    seg_x, carried = carried_sets(founders, extent, chromosome_map, total)
    if config.sites.variant_against == "reference":
        carried[:, -1] = True
    length = float(relative_morgans(extent, chromosome_map, np.array([extent.end_bp]))[0])
    rng = stock_stream("simulation.source", seed, contig)
    saved = copy.deepcopy(rng.bit_generator.state)
    args = (leaves, parent, split, size, length)
    starts, separating, _ = separating_length(*args, rng, seg_x, carried)
    # Rounded no-map stock crossovers can land exactly at the terminal bp.
    # R5's final interval includes that coordinate. If C changes there, replay
    # the same trees with terminal C and append its point interval for the bp
    # clock calculation; mutation placement uses that same terminal replay.
    terminal = np.zeros(total + 1, bool)
    for pair in founders:
        for haplotype in pair:
            terminal[int(source_at(haplotype, np.array([extent.end_bp]))[0])] = True
    if config.sites.variant_against == "reference":
        terminal[-1] = True
    terminal_changed = not np.array_equal(terminal, carried[-1])
    terminal_segments = np.array([0.0, length])
    if terminal_changed:
        rng.bit_generator.state = copy.deepcopy(saved)
        end_starts, end_separating, _ = separating_length(*args, rng, terminal_segments, terminal[None, :])
        if starts[-1] == length:
            separating = separating.copy()
            separating[-1] = end_separating[-1]
        else:
            starts = np.r_[starts, length]
            separating = np.r_[separating, end_separating[-1]]
    profile = realise_profile(
        extent, config.sites.profile.parameters(), stock_stream("simulation.profile", seed, contig)
    )
    site_rng = stock_stream("simulation.sites", seed, contig)
    positions, spurious, dropped = draw_site_set(
        extent,
        _GeneratingMap(chromosome_map),
        starts,
        separating,
        profile,
        config.sites.mode(int(extent.end_bp)),
        config.sites.spurious_fraction,
        site_rng,
    )
    rng.bit_generator.state = saved
    gene_positions = positions[~spurious]
    coordinates = relative_morgans(extent, chromosome_map, gene_positions)
    uniforms = site_rng.random(len(gene_positions))
    if terminal_changed and len(gene_positions) and gene_positions[-1] == extent.end_bp:
        derived = np.empty((total + 1, len(gene_positions)), np.int8)
        derived[:, :-1] = mutations(*args, rng, seg_x, carried, coordinates[:-1], uniforms[:-1])
        rng.bit_generator.state = copy.deepcopy(saved)
        derived[:, -1:] = mutations(*args, rng, terminal_segments, terminal[None, :], coordinates[-1:], uniforms[-1:])
    else:
        derived = mutations(*args, rng, seg_x, carried, coordinates, uniforms)
    if np.any(spurious):
        all_derived = np.zeros((total + 1, len(positions)), np.int8)
        all_derived[:, ~spurious] = derived
        derived = all_derived
    reference_derived = derived[-1].astype(bool)
    ids = np.unique(np.concatenate([source_at(h, positions) for pair in founders for h in pair])).astype(np.int32)
    coded = derived[ids] ^ derived[-1]
    return SourcePanel(positions, coded, ids, populations[ids], reference_derived, total, spurious), dropped


def _contig_batches(active, function, batches):
    """Replay the released seed schedule, computing only the active contig's genetics.

    Other axis entries carry a one-segment placeholder. Their child seed blocks are
    still drawn by pedigree.py, exactly as in the multi-contig released simulator.
    """
    results = []
    for batch in batches:
        c, first, pairs = batch[:3]
        if c == active:
            results.append(function(batch))
        else:
            placeholder = batch[4][0]
            children = [(placeholder, placeholder, [np.empty(0), np.empty(0)]) for _ in pairs]
            results.append((c, first, children))
    return results


def streamed_cross(config, seed, maps, rows, founder_count, *, active=None, founders=None, positions=None,
                   structure=None):
    """One actual contig, a full released cross seed axis, and shared pedigree metadata."""
    placeholder = [(founder_mosaic(1.0, 0, 2 * i), founder_mosaic(1.0, 0, 2 * i + 1)) for i in range(founder_count)]
    axis = [placeholder] * len(config.contigs)
    sites = [np.array([1], np.int64)] * len(config.contigs)
    if active is not None:
        axis[active], sites[active] = founders, positions
    common = dict(
        seed=seed,
        recombination_rate=maps[0].fallback_rate_per_bp,
        mutation_rate=config.mutation_rate_per_bp,
        chromosome_maps=[m if m.has_map else None for m in maps],
        budget=1,
        runner=partial(_contig_batches, active),
    )
    if structure is not None:
        common["structures"] = [structure if c == active else None for c in range(len(config.contigs))]
    if rows is None:
        generated, simulated = simulate_pedigree(axis, sites, config.cohorts, design=config.design, **common)
        selected = observed_indices(generated.cohorts, config.design, seed)
    else:
        generated, simulated = simulate_explicit_pedigree(axis, sites, rows, **common)
        selected = np.array([i for i, row in enumerate(rows) if row.observed], np.int64)
    return generated, generated.take(selected), selected, None if active is None else simulated[active]


def prepare_contig(config, seed, index, maps, rows, records, order, template=None):
    """Generate one source/stock/cross contig without retaining any other site's arrays."""
    contig, chromosome_map = config.contigs[index], maps[index]
    extent = (
        Extent(1.0, float(config.contig_length(contig)))
        if config.source.kind == "coalescent"
        else Extent(float(template.positions[0]), float(template.positions[-1]))
    )
    dropped = np.empty(0, np.int64)
    panel = None
    if config.source.kind != "coalescent":
        panel = template.materialise(stream("simulation.founders", seed=seed, index=index))
    structure = _contig_structure(config, seed, contig, extent, chromosome_map, records, panel)
    if records:
        founders = cross_founders(config, records, order, extent, chromosome_map, seed, contig, structure)
    if config.source.kind == "coalescent":
        panel, dropped = coalescent_panel(config, records, founders, extent, chromosome_map, seed, contig)
    elif not records:
        founders = pair_founders([founder_mosaic(extent.start_bp, int(h), int(h)) for h in panel.source_id])
    collapsed = draw_collapsed(
        config.structure, seed, contig, int(extent.start_bp), int(extent.end_bp), int(panel.total_sources)
    )
    if collapsed is not None:
        panel = _with_psv_sites(panel, collapsed_psv_sites(collapsed, panel.positions))
    # The generated extent is [1, length_bp], independently of its retained sites.
    # A terminal sentinel keeps the released no-map traversal on the whole extent.
    # Scalar clock origin and actual marker coordinates remain separate.
    cross_positions = panel.positions
    if config.source.kind == "coalescent":
        cross_positions = _CrossSites(np.unique(np.r_[panel.positions, int(extent.end_bp)]), extent.start_bp)
    generated, observed, selected, simulated = streamed_cross(
        config, seed, maps, rows, len(founders), active=index, founders=founders, positions=cross_positions,
        structure=structure,
    )
    offspring = [simulated.haplotypes[row] for row in selected]
    # Contamination donors that are not sequenced (explicit pedigrees): their haplotypes travel with the task.
    observed_names = set(map(str, observed.samples))
    generated_names = [str(s) for s in generated.samples]
    donor_ids = tuple(dict.fromkeys(d for d in config.reads.contamination_donors if d not in observed_names))
    missing = [d for d in donor_ids if d not in generated_names]
    if missing:
        raise SettingsError(f"contamination_donors name individuals the simulation does not generate: {missing[:5]}")
    donor_rows = [generated_names.index(d) for d in donor_ids]
    n_offspring = len(offspring)
    offspring = offspring + [simulated.haplotypes[row] for row in donor_rows]
    if config.source.kind == "coalescent":
        retained = np.searchsorted(cross_positions, panel.positions)
        translated = []
        for pair in offspring:
            members = []
            for haplotype in pair:
                flips = np.flatnonzero(np.isin(retained, haplotype.flips)).astype(np.int64)
                # Spurious records describe no variant in any biological haplotype.
                # Keep the released meiosis draws; their reported flips at these
                # error-only records are absent from the observation truth.
                if np.any(panel.spurious):
                    flips = flips[~panel.spurious[flips]]
                members.append(Mosaic(haplotype.starts, haplotype.source, haplotype.founder, flips))
            translated.append(tuple(members))
        offspring = translated
    elif collapsed is not None and np.any(panel.spurious):
        # A PSV site is reference-only in every haplotype: drop meiotic mutations there, as at spurious sites.
        reference_only = np.flatnonzero(panel.spurious)
        offspring = [
            tuple(Mosaic(h.starts, h.source, h.founder, h.flips[~np.isin(h.flips, reference_only)]) for h in pair)
            for pair in offspring
        ]
    donor_haplotypes = tuple(offspring[n_offspring:])
    offspring = offspring[:n_offspring]
    analysed = {int(row): i for i, row in enumerate(selected)}
    return ContigSimulation(
        contig,
        observed.samples,
        panel.positions,
        offspring,
        offspring,
        founders,
        panel,
        truth_crossovers(simulated.gametes, analysed, generated.samples, generated.cohorts),
        config.depth,
        config.reads,
        read_seeds(seed, len(config.contigs))[index],
        seed,
        config.artefacts,
        dropped,
        structure,
        None if structure is None else _switch_truth(simulated, analysed),
        collapsed,
        donor_ids,
        donor_haplotypes,
    )


def _with_psv_sites(panel: SourcePanel, psv: NDArray[np.bool_]) -> SourcePanel:
    """The panel with its PSV sites reference-only in every source haplotype and marked spurious (structure.py)."""
    alleles = np.array(panel.alleles, copy=True)
    alleles[:, psv] = 0
    return replace(
        panel,
        alleles=alleles,
        reference_derived=np.asarray(panel.reference_derived, bool) & ~psv,
        spurious=np.asarray(panel.spurious, bool) | psv,
    )


def _switch_truth(simulated, analysed):
    """Reference-axis switches of the analysed children's gametes, in truth_crossovers order."""
    parts = [
        np.asarray(simulated.switches[(row, slot)], np.float64)
        for row, slot, _, _ in simulated.gametes
        if row in analysed
    ]
    offsets = np.concatenate(([0], np.cumsum([len(x) for x in parts]))).astype(np.int64)
    return offsets, (np.concatenate(parts) if parts else np.empty(0, np.float64))


def _contig_structure(config, seed, contig, extent, chromosome_map, records, panel):
    """The contig's structural variants (simulation/structure.py), or None when the model is off.

    Carriers: with a coalescent source, the derived state of a mutation on the source genealogy at each variant's
    anchor, replayed on the same trees as the sites (the source stream restarted; the traversal's draws do not depend
    on the carried sets); with a file source, independent draws per source haplotype."""
    model = config.structure
    if not model.enabled:
        return None
    first, last = int(extent.start_bp), int(extent.end_bp)
    if config.source.kind == "coalescent":
        names, parent, split, size = config.source.population_tree()
        populations = np.concatenate(
            [np.full(2 * int(r.sizes[0]), names.index(r.population), np.int64) for r in records]
        )
        reference = names.index(config.source.reference_population) if config.source.reference_population else 0
        leaves = np.r_[populations, np.int64(reference)]
        length = float(relative_morgans(extent, chromosome_map, np.array([extent.end_bp]))[0])

        def genealogy(anchors, uniforms):
            order = np.argsort(anchors, kind="stable")
            x = np.clip(relative_morgans(extent, chromosome_map, np.asarray(anchors, np.float64)[order]), 0, length)
            rng = stock_stream("simulation.source", seed, contig)
            derived = mutations(
                leaves, parent, split, size, length, rng, np.array([0.0, length]),
                np.ones((1, len(leaves)), bool), x, np.asarray(uniforms, np.float64)[order],
            )
            result = np.empty_like(derived)
            result[:, order] = derived
            return result

        return draw_structure(model, seed, contig, first, last, len(populations), genealogy)
    return draw_structure(model, seed, contig, first, last, len(panel.source_id))