"""Generate a simulation's typed contig inputs without storage or scheduling effects.

Owns founder materialisation, observation selection and the reads/truth transition
(OU23, OU29, OU30; docs/methods/simulation.md). The stage binds the realised seed
and supplies templates, maps and explicit pedigree rows. It publishes the results.

Draw order: founders use SeedSequence([seed, 2000000]).spawn(C)[c]; pedigree
uses the master stream (one initial Python structure seed for random cohorts only),
then a contig-major child-seed block per cohort/depth. Observation uses its separate
SeedSequence stream. Read seeds are drawn for EVERY contig before any resume skip;
each read worker consumes its whole [N,L] tensor, never sample chunks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..contracts.genetic_map import ChromosomeGeneticMap
from ..contracts.roles import freeze_record_arrays, record_field
from ..contracts.source import ContigReads
from ..core.errors import InputError
from ..core.random import read_seeds, stream
from .design import PedigreeRow, ReadModel, SimulationConfig, observed_indices
from .pedigree import PedigreeTable, materialise_founders, pair_founders, simulate_explicit_pedigree, simulate_pedigree
from .reads import sample_reads
from .truth import ContigTruth, TruthCrossovers, truth_crossovers, truth_painting

if TYPE_CHECKING:
    from numpy.typing import NDArray

    # ContigReads.__post_init__ accepts these int64 draws and freezes them as int32.
    _ContigReadsFactory = Callable[[str, tuple[str, ...], NDArray[np.int64], NDArray[np.int64]], ContigReads]

__all__ = ["FounderTemplate", "ContigSimulation", "SimulationPlan", "prepare_simulation", "generate_contig"]


@dataclass(frozen=True)
class FounderTemplate:
    """One input template: int64[L] positions and float64[H,L,2] probabilities.

    H is even and the same for every contig. Arrays are borrowed and never mutated.
    Validation precedes all draws; invalid user templates raise InputError.
    """

    positions: np.ndarray = record_field("obs", axes=("L",), units="bp", dtype="int64")
    allele_probabilities: np.ndarray = record_field("obs", axes=("H", "L", "2"), units="probability", dtype="float64")

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.int64)
        probabilities = np.asarray(self.allele_probabilities)
        if positions.ndim != 1:
            raise InputError("founder template positions must have one site axis")
        if probabilities.ndim != 3 or probabilities.shape[1:] != (len(positions), 2):
            raise InputError("founder template axes must be (haplotypes, sites, 2)")
        if len(positions) == 0 or np.any(np.diff(positions) <= 0):
            raise InputError("founder template positions must be non-empty and increasing")
        if len(probabilities) % 2:
            raise InputError("founder templates must hold an even number of haplotypes")
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities.sum(axis=2) <= 0):
            raise InputError("founder template probabilities must be finite, nonnegative, with positive mass")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "allele_probabilities", probabilities)
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ContigSimulation:
    """A whole-contig reads task with its already drawn seed and observed truth.

    Offspring is [N][2] int8[L]; ancestry is [N][2] lists of float-bp segments.
    No field is mutated by generate_contig. Splitting the sample axis is illegal:
    it changes the generating read sampler's draw order.
    """

    contig: str = record_field("obs", axes=(), units="1", dtype="str")
    sample_ids: tuple[str, ...] = record_field("obs", axes=(), units="1", dtype="tuple[str, ...]")
    positions: np.ndarray = record_field("obs", axes=(), units="1", dtype="np.ndarray")
    offspring: list = record_field("obs", axes=(), units="1", dtype="list")
    ancestry: list = record_field("obs", axes=(), units="1", dtype="list")
    founders: list = record_field("obs", axes=(), units="1", dtype="list")
    crossovers: TruthCrossovers = record_field("obs", axes=(), units="1", dtype="TruthCrossovers")
    depth: float = record_field("obs", axes=(), units="1", dtype="float")
    read_model: ReadModel = record_field("obs", axes=(), units="1", dtype="ReadModel")
    seed: int = record_field("obs", axes=(), units="1", dtype="int")


@dataclass(frozen=True)
class SimulationPlan:
    """Shared pedigree truth and contig tasks in the configured contig order."""

    generated: PedigreeTable = record_field("obs", axes=(), units="1", dtype="PedigreeTable")
    observed: PedigreeTable = record_field("obs", axes=(), units="1", dtype="PedigreeTable")
    contigs: tuple[ContigSimulation, ...] = record_field(
        "obs", axes=(), units="1", dtype="tuple[ContigSimulation, ...]"
    )


def prepare_simulation(
    config: SimulationConfig,
    seed: int,
    templates: Sequence[FounderTemplate],
    chromosome_maps: Sequence[ChromosomeGeneticMap],
    *,
    pedigree_rows: tuple[PedigreeRow, ...] | None = None,
    cores: int = 1,
    runner: Callable | None = None,
) -> SimulationPlan:
    """Materialise founders and shared pedigree; return tasks with all read seeds bound.

    Templates/maps follow config.contigs order. A supplied seed is already realised.
    Explicit rows override cohorts and retain table observation. Input axes are not
    mutated; unequal founder counts across contigs raise InputError before draws.
    The runner executes independent child batches and returns indexed results.
    """
    if len(templates) != len(config.contigs) or len(chromosome_maps) != len(config.contigs):
        raise InputError("templates and generating maps must follow the configured contig axis")
    if len({len(t.allele_probabilities) for t in templates}) != 1:
        raise InputError("simulation contigs have different founder counts")
    sites = [t.positions for t in templates]
    founders = [
        pair_founders(materialise_founders(t.allele_probabilities, stream("simulation.founders", seed=seed, index=c)))
        for c, t in enumerate(templates)
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
            truth_crossovers(simulated[c].gametes, analysed, generated.samples, generated.cohorts),
            config.depth,
            config.reads,
            seeds[c],
        )
        for c, contig in enumerate(config.contigs)
    )
    return SimulationPlan(generated, observed, tasks)


def generate_contig(task: ContigSimulation) -> tuple[ContigReads, ContigTruth]:
    """Return reads int32[N,L,2] and unchanged truth for one whole contig.

    Maternal and paternal arrays preserve the observed pedigree row order. Mutation,
    map crossover positions and integer truth-painting truncation are retained.
    The only mutable object is the task-local PCG64 stream.
    """
    rng = stream("simulation.contig_reads", seed=task.seed)
    hap0 = np.array([individual[0] for individual in task.offspring])
    hap1 = np.array([individual[1] for individual in task.offspring])
    counts = sample_reads(hap0, hap1, task.positions, task.depth, task.read_model, rng)
    truth = ContigTruth(
        sample_ids=task.sample_ids,
        positions=task.positions,
        founder_haplotypes=np.asarray([h for pair in task.founders for h in pair], dtype=np.int8),
        alleles=np.ascontiguousarray(np.stack((hap0, hap1), axis=2), dtype=np.int8),
        painting=truth_painting(task.ancestry),
        crossovers=task.crossovers,
    )
    return cast("_ContigReadsFactory", ContigReads)(task.contig, task.sample_ids, task.positions, counts), truth
