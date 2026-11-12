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
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..contracts.genetic_map import ChromosomeGeneticMap
from ..contracts.roles import freeze_record_arrays, record_field
from ..contracts.source import ContigReads
from ..core.errors import InputError
from ..core.random import read_seeds, stream
from .artefacts import ArtefactModel, ArtefactResult, apply_artefacts
from .design import PedigreeRow, ReadModel, SimulationConfig, observed_indices
from .genealogy import mutations, separating_length
from .haplotypes import Extent, Mosaic, alleles, founder_mosaic, source_at
from .pedigree import PedigreeTable, materialise_founders, pair_founders, simulate_explicit_pedigree, simulate_pedigree
from .reads import ReadInputs, read_inputs
from .sites import draw_site_set, realise_profile, relative_morgans
from .stocks import stock_founders, stock_record, stock_stream
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
    return read_inputs(hap0, hap1, task.positions, task.depth, task.read_model, rng)


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
        )
    else:
        transformed = ArtefactResult(counts, None, None, None)
    truth = ContigTruth(
        sample_ids=task.sample_ids,
        positions=task.positions,
        alleles=np.ascontiguousarray(np.stack((inputs.hap0, inputs.hap1), axis=2), dtype=np.int8),
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
        block=transformed.block,
        systematic_site=transformed.systematic_site,
        spurious=task.panel.spurious,
    )
    return cast("_ContigReadsFactory", ContigReads)(
        task.contig, task.sample_ids, task.positions, transformed.counts
    ), truth


def prepare_stocks(config, seed, rows):
    """Stable source layout and shared pedigrees; all capacities have been validated."""
    order = config.founder_stocks(rows)
    records, first = [], 0
    for specification in config.stocks:
        records.append(stock_record(specification.name, specification, order.count(specification.name), first, seed))
        first += 2 * specification.founders
    return tuple(records), order


def cross_founders(config, records, order, extent, chromosome_map, seed, contig):
    """Take whole individuals in pedigree order, then assign the cross-founder axis."""
    taken = {r.name: iter(stock_founders(r, extent, chromosome_map, seed, contig)) for r in records}
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


def streamed_cross(config, seed, maps, rows, founder_count, *, active=None, founders=None, positions=None):
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
    if records:
        founders = cross_founders(config, records, order, extent, chromosome_map, seed, contig)
    if config.source.kind == "coalescent":
        panel, dropped = coalescent_panel(config, records, founders, extent, chromosome_map, seed, contig)
    else:
        panel = template.materialise(stream("simulation.founders", seed=seed, index=index))
        if not records:
            founders = pair_founders([founder_mosaic(extent.start_bp, int(h), int(h)) for h in panel.source_id])
    # The generated extent is [1, length_bp], independently of its retained sites.
    # A terminal sentinel keeps the released no-map traversal on the whole extent.
    # Scalar clock origin and actual marker coordinates remain separate.
    cross_positions = panel.positions
    if config.source.kind == "coalescent":
        cross_positions = _CrossSites(np.unique(np.r_[panel.positions, int(extent.end_bp)]), extent.start_bp)
    generated, observed, selected, simulated = streamed_cross(
        config, seed, maps, rows, len(founders), active=index, founders=founders, positions=cross_positions
    )
    offspring = [simulated.haplotypes[row] for row in selected]
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
    )
