"""Founders, pedigree structure and meiosis of a known-pedigree simulation.

Random streams (fixed so that a configuration and seed always give the same data, at any worker
count, and the same data as the base release 92b3daf):

* founder haplotypes of contig c: ``default_rng(SeedSequence([seed, 2_000_000]).spawn(C)[c])``
  resolves exact probability ties of the templates (``materialise_founders``);
* ``master = default_rng(seed)``; the pedigree structure draws from a Mersenne Twister
  ``random.Random(master.integers(0, 2**31))`` (parent pairs, backcross choices);
* per cohort, ``master.integers(0, 2**63, size=(C, cohort size))`` gives one seed per contig and
  child, drawn before any scheduling; each gamete of that child on that contig then draws from
  ``default_rng(child seed)``: which parental homologue starts, the crossovers, the mutations.

Crossovers: without a generating map (or on a contig the map lacks), gaps between crossovers are
exponential with the per-bp rate, rounded up to whole bp; with a map, a Poisson number of
crossovers uniform in genetic distance. Ancestry segments (start, end, founder haplotype) follow
every crossover; mutations flip alleles at a per-bp rate (exponential gaps, rounded down).
"""
from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from ..execution.budget import as_budget
from ..execution.pools import BudgetedPool
from .design import PedigreeRow, SimulationDesign


# ---------------------------------------------------------------------------------------------
# Founders
# ---------------------------------------------------------------------------------------------

def materialise_founders(allele_probabilities, rng: np.random.Generator) -> list[np.ndarray]:
    """Concrete founder haplotypes (int8 0/1) from template probabilities float[H, L, 2].

    Each site takes its most probable allele; exact ties (including unknown (0.5, 0.5) cells) are
    resolved by ``rng``, haplotype by haplotype, so no reference bias enters the truth.
    """
    result = []
    for haplotype in allele_probabilities:
        values = np.asarray(haplotype)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("founder template haplotypes must have shape (sites, 2)")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(np.sum(values, axis=1) <= 0.0):
            raise ValueError("founder template probabilities must be finite, nonnegative, with positive mass")
        alleles = np.argmax(values, axis=1).astype(np.int8)
        tied = values[:, 0] == values[:, 1]
        if np.any(tied):
            alleles[tied] = rng.integers(0, 2, size=int(np.sum(tied)), dtype=np.int8)
        result.append(alleles)
    return result


def pair_founders(haplotypes) -> list[list[np.ndarray]]:
    """Diploid founders from consecutive haplotypes (0, 1), (2, 3), ..."""
    if len(haplotypes) % 2:
        raise ValueError("founder templates must hold an even number of haplotypes")
    return [[haplotypes[2 * i], haplotypes[2 * i + 1]] for i in range(len(haplotypes) // 2)]


# ---------------------------------------------------------------------------------------------
# Meiosis
# ---------------------------------------------------------------------------------------------

def _segments_in_range(painting, range_start, range_end):
    """Ancestry segments (start, end, founder) of ``painting`` clipped to [range_start, range_end)."""
    result = []
    for seg_start, seg_end, founder in painting:
        overlap_start = max(seg_start, range_start)
        overlap_end = min(seg_end, range_end)
        if overlap_start < overlap_end:
            result.append((overlap_start, overlap_end, founder))
    return result


def _map_crossovers(site_locs, chromosome_map, rng):
    """A Poisson process uniform in integrated genetic coordinates over the contig's sites."""
    start, end = float(site_locs[0]), float(site_locs[-1])
    distance = float(chromosome_map.interval_morgans(start, end))
    if distance == 0:
        return np.empty(0, dtype=np.float64)
    count = int(rng.poisson(distance))
    genetic_start = float(chromosome_map.cumulative_morgans(start))
    coordinates = genetic_start + rng.uniform(0., distance, size=count)
    return np.sort(chromosome_map.inverse_morgans(coordinates))


def gamete(hap_pair, ancestry_pair, site_locs, *, recombination_rate, mutation_rate, rng,
           chromosome_map=None):
    """One gamete of a parent: (alleles int8[L], ancestry segments, crossover positions float64[]).

    ``chromosome_map`` (a ChromosomeGeneticMap with a map) replaces the homogeneous
    ``recombination_rate`` (per bp). Crossover positions are in bp (float).
    """
    using_map = chromosome_map is not None and chromosome_map.has_map
    start_phys = site_locs[0]
    end_phys = site_locs[-1]
    cur_loc = start_phys
    cur_loc_index = 0
    using_hap = rng.choice([0, 1])
    alleles, ancestry, crossovers = [], [], []
    if not using_map and recombination_rate > 0:
        # Exponential gaps rounded up to whole bp, from the first site until past the last.
        recombination_scale = 1.0 / recombination_rate
        while cur_loc <= end_phys:
            new_loc = cur_loc + np.ceil(rng.exponential(recombination_scale))
            new_loc_index = np.searchsorted(site_locs, new_loc)
            alleles.append(hap_pair[using_hap][cur_loc_index:new_loc_index])
            ancestry.extend(_segments_in_range(ancestry_pair[using_hap], cur_loc, min(new_loc, end_phys)))
            if new_loc <= end_phys:
                crossovers.append(float(new_loc))
            using_hap = 1 - using_hap
            cur_loc = new_loc
            cur_loc_index = new_loc_index
            if cur_loc_index >= len(site_locs):
                break
    else:
        events = (_map_crossovers(site_locs, chromosome_map, rng) if using_map
                  else np.empty(0, dtype=np.float64))
        # A terminal boundary beyond the last marker includes the final allele while keeping the
        # [start, end) ancestry convention.
        terminal_loc = np.nextafter(float(end_phys), np.inf)
        for event_index in range(len(events) + 1):
            is_crossover = event_index < len(events)
            new_loc = events[event_index] if is_crossover else terminal_loc
            new_loc_index = np.searchsorted(site_locs, new_loc)
            alleles.append(hap_pair[using_hap][cur_loc_index:new_loc_index])
            ancestry.extend(_segments_in_range(ancestry_pair[using_hap], cur_loc, min(new_loc, end_phys)))
            if is_crossover:
                crossovers.append(float(new_loc))
                using_hap = 1 - using_hap
                cur_loc = new_loc
                cur_loc_index = new_loc_index
    result = np.concatenate(alleles)[:len(site_locs)]
    # Mutations flip alleles (ancestry unchanged): exponential gaps rounded down.
    mutation_scale = 1.0 / mutation_rate
    mutated = []
    cur_loc = start_phys
    while cur_loc <= end_phys:
        new_loc = cur_loc + np.floor(rng.exponential(mutation_scale))
        new_loc_index = np.searchsorted(site_locs, new_loc)
        if new_loc_index < len(site_locs):
            mutated.append(new_loc_index)
        cur_loc = new_loc
    if mutated:
        result[mutated] = 1 - result[mutated]
    return result, ancestry, np.asarray(crossovers, dtype=np.float64)


def _simulate_children(task, shared=None, budget=None):
    """Pool task: the children ``first..stop-1`` of one contig and cohort (one seed per child)."""
    (contig_index, first, parent_pairs, seeds, parents, ancestries, site_locs, recombination_rate,
     mutation_rate, chromosome_map) = task
    children = []
    for (mother, father), seed in zip(parent_pairs, seeds):
        rng = np.random.default_rng(int(seed))
        maternal = gamete(parents[mother], ancestries[mother], site_locs,
                          recombination_rate=recombination_rate, mutation_rate=mutation_rate,
                          rng=rng, chromosome_map=chromosome_map)
        paternal = gamete(parents[father], ancestries[father], site_locs,
                          recombination_rate=recombination_rate, mutation_rate=mutation_rate,
                          rng=rng, chromosome_map=chromosome_map)
        children.append(([maternal[0], paternal[0]], [maternal[1], paternal[1]],
                         [maternal[2], paternal[2]]))
    return contig_index, first, children


# ---------------------------------------------------------------------------------------------
# Pedigree
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PedigreeTable:
    """Individuals in generation order: sample names, cohort (1 = children of the founders) and
    the two parents (founders are ``Founder_<i>``); parent1 gave the first haplotype."""

    samples: tuple[str, ...]
    cohorts: tuple[int, ...]
    parent1: tuple[str, ...]
    parent2: tuple[str, ...]

    def take(self, rows) -> "PedigreeTable":
        rows = [int(row) for row in rows]
        return PedigreeTable(*(tuple(column[row] for row in rows) for column in
                               (self.samples, self.cohorts, self.parent1, self.parent2)))

    def record(self) -> dict:
        return dict(samples=list(self.samples), cohorts=list(self.cohorts),
                    parent1=list(self.parent1), parent2=list(self.parent2))

    @classmethod
    def from_record(cls, record) -> "PedigreeTable":
        return cls(tuple(record["samples"]), tuple(int(c) for c in record["cohorts"]),
                   tuple(record["parent1"]), tuple(record["parent2"]))


@dataclass
class SimulatedContig:
    """The generated individuals of one contig, in pedigree order (one entry per individual).

    haplotypes: [maternal, paternal] int8[L] alleles; ancestry: per haplotype, segments (start,
    end, founder haplotype index); gametes: per gamete, (child row, parent slot 0/1, parent name,
    crossover positions float64[]).
    """

    haplotypes: list
    ancestry: list
    gametes: list


class _TaskRunner:
    """A BudgetedPool for more than one process; inline execution (no process) for one."""

    def __init__(self, budget, processes):
        self.pool = (BudgetedPool(budget, processes=processes, label="simulated meiosis")
                     if processes > 1 else None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            return self.pool.__exit__(*exc)
        return False

    def imap_unordered(self, function, tasks):
        if self.pool is None:
            return (function(task, None, None) for task in tasks)
        return self.pool.imap_unordered(function, tasks)


def _children_per_task(n_children: int, n_contigs: int, processes: int) -> int:
    """Children per pool task: as base, each contig's cohort is split over ~processes / contigs tasks."""
    per_contig = max(1, processes // max(1, min(n_contigs, processes)))
    return max(1, (n_children + per_contig - 1) // per_contig)


def simulate_pedigree(founders_by_contig, sites_by_contig, cohort_sizes, *, seed: int,
                      recombination_rate: float, mutation_rate: float, chromosome_maps,
                      design: SimulationDesign, budget):
    """Simulate the cohorts on every contig with one shared pedigree structure.

    founders_by_contig[c]: list of founders [hap0, hap1] (int8[L_c]); sites_by_contig[c]: int64[L_c];
    chromosome_maps[c]: the contig's generating ChromosomeGeneticMap (None: the homogeneous rate).
    Returns (PedigreeTable of every generated individual, [SimulatedContig per contig]).
    """
    budget = as_budget(budget)
    n_contigs = len(founders_by_contig)
    master = np.random.default_rng(seed)
    structure = random.Random(int(master.integers(0, 2 ** 31)))

    # Founder ancestry: haplotype h of the founders covers the whole contig.
    parents_by_contig = [list(founders) for founders in founders_by_contig]
    ancestries_by_contig = []
    for founders, sites in zip(founders_by_contig, sites_by_contig):
        start, end = sites[0], sites[-1]
        ancestries_by_contig.append([[[(start, end, 2 * i)], [(start, end, 2 * i + 1)]]
                                     for i in range(len(founders))])
    parent_ids = [f"Founder_{i}" for i in range(len(founders_by_contig[0]))]
    contigs = [SimulatedContig([], [], []) for _ in range(n_contigs)]
    samples, cohorts, parent1, parent2 = [], [], [], []
    # Backcrosses pair a generated individual with one of its actual parents.
    parents_of = {}
    hap_registry = [dict(zip(parent_ids, parents)) for parents in parents_by_contig]
    ancestry_registry = [dict(zip(parent_ids, ancestries)) for ancestries in ancestries_by_contig]

    processes = budget.threads()
    with _TaskRunner(budget, processes) as pool:
        for cohort_index, n_children in enumerate(cohort_sizes):
            cohort_name = f"F{cohort_index + 1}"
            backcrossing = (design.backcross_fraction > 0
                            and cohort_index + 1 >= design.backcross_start_cohort)
            pool_ids = list(parent_ids)
            if backcrossing:
                for individual in parent_ids:
                    for parent in parents_of[individual]:
                        if parent not in pool_ids:
                            pool_ids.append(parent)
            pool_index = {name: index for index, name in enumerate(pool_ids)}
            pool_haplotypes = ([[registry[name] for name in pool_ids] for registry in hap_registry]
                               if backcrossing else parents_by_contig)
            pool_ancestries = ([[registry[name] for name in pool_ids] for registry in ancestry_registry]
                               if backcrossing else ancestries_by_contig)
            # A. Structure (shared by every contig).
            parent_pairs, child_ids = [], []
            for i in range(n_children):
                if backcrossing and structure.random() < design.backcross_fraction:
                    first_id = structure.choice(parent_ids)
                    second_id = structure.choice(parents_of[first_id])
                    pair = (pool_index[first_id], pool_index[second_id])
                else:
                    pair = tuple(structure.sample(range(len(parent_ids)), 2))
                    first_id, second_id = pool_ids[pair[0]], pool_ids[pair[1]]
                child_id = f"{cohort_name}_{i}"
                parents_of[child_id] = (first_id, second_id)
                child_ids.append(child_id)
                parent_pairs.append(pair)
                samples.append(child_id)
                cohorts.append(cohort_index + 1)
                parent1.append(first_id)
                parent2.append(second_id)
            # B. Genetics: one seed per contig and child, fixed before any scheduling.
            seeds = master.integers(0, 2 ** 63, size=(n_contigs, n_children), dtype=np.int64)
            per_task = _children_per_task(n_children, n_contigs, processes)
            tasks = [(c, first, parent_pairs[first:first + per_task], seeds[c, first:first + per_task],
                      pool_haplotypes[c], pool_ancestries[c], sites_by_contig[c], recombination_rate,
                      mutation_rate, chromosome_maps[c])
                     for c in range(n_contigs) for first in range(0, n_children, per_task)]
            offspring = [[None] * n_children for _ in range(n_contigs)]
            for contig_index, first, children in pool.imap_unordered(_simulate_children, tasks):
                offspring[contig_index][first:first + len(children)] = children
            row0 = len(samples) - n_children
            for c in range(n_contigs):
                for i, (haplotypes, ancestry, events) in enumerate(offspring[c]):
                    contigs[c].haplotypes.append(haplotypes)
                    contigs[c].ancestry.append(ancestry)
                    for slot in (0, 1):
                        parent = pool_ids[parent_pairs[i][slot]]
                        contigs[c].gametes.append((row0 + i, slot, parent, events[slot]))
            if design.backcross_fraction:
                for c in range(n_contigs):
                    hap_registry[c].update(zip(child_ids, (child[0] for child in offspring[c])))
                    ancestry_registry[c].update(zip(child_ids, (child[1] for child in offspring[c])))
            parents_by_contig = [[child[0] for child in offspring[c]] for c in range(n_contigs)]
            ancestries_by_contig = [[child[1] for child in offspring[c]] for c in range(n_contigs)]
            parent_ids = child_ids
    return PedigreeTable(tuple(samples), tuple(cohorts), tuple(parent1), tuple(parent2)), contigs


def simulate_explicit_pedigree(founders_by_contig, sites_by_contig, rows: tuple[PedigreeRow, ...], *, seed: int,
                               recombination_rate: float, mutation_rate: float, chromosome_maps, budget):
    """Simulate an explicit pedigree table (design.read_pedigree_table rows) on every contig (release 2.17).

    Founder rows take the template founders in table order (their numbers must agree). Children are simulated by
    depth (a founder has depth 0, a child one more than its deeper parent); each depth draws from the master stream
    as one cohort of the random design does: one seed per contig and child, drawn before any scheduling, and each
    gamete of that child on that contig from ``default_rng(child seed)``. Returns (PedigreeTable of every individual
    in table order, founders included with cohort 0 and empty parents, cohort = depth otherwise; [SimulatedContig]
    per contig in the same order; a founder's haplotypes are its template founder, its ancestry its own two
    founder haplotypes, and it has no gametes).
    """
    budget = as_budget(budget)
    n_contigs = len(founders_by_contig)
    founder_rows = [index for index, row in enumerate(rows) if not row.parent1]
    if len(founder_rows) != len(founders_by_contig[0]):
        raise ValueError(f"the pedigree table has {len(founder_rows)} founders and the templates "
                         f"{len(founders_by_contig[0])}")
    master = np.random.default_rng(seed)
    depth = {}
    for row in rows:
        depth[row.sample] = 0 if not row.parent1 else 1 + max(depth[row.parent1], depth[row.parent2])
    index_of = {row.sample: index for index, row in enumerate(rows)}
    n = len(rows)
    contigs = [SimulatedContig([None] * n, [None] * n, []) for _ in range(n_contigs)]
    for founder, index in enumerate(founder_rows):
        for c in range(n_contigs):
            start, end = sites_by_contig[c][0], sites_by_contig[c][-1]
            contigs[c].haplotypes[index] = list(founders_by_contig[c][founder])
            contigs[c].ancestry[index] = [[(start, end, 2 * founder)], [(start, end, 2 * founder + 1)]]
    processes = budget.threads()
    with _TaskRunner(budget, processes) as pool:
        for level in range(1, max(depth.values()) + 1):
            members = [index for index, row in enumerate(rows) if depth[row.sample] == level]
            ancestors = sorted({index_of[parent] for index in members
                                for parent in (rows[index].parent1, rows[index].parent2)})
            slot_of = {index: slot for slot, index in enumerate(ancestors)}
            parent_pairs = [(slot_of[index_of[rows[index].parent1]], slot_of[index_of[rows[index].parent2]])
                            for index in members]
            seeds = master.integers(0, 2 ** 63, size=(n_contigs, len(members)), dtype=np.int64)
            per_task = _children_per_task(len(members), n_contigs, processes)
            tasks = [(c, first, parent_pairs[first:first + per_task], seeds[c, first:first + per_task],
                      [contigs[c].haplotypes[index] for index in ancestors],
                      [contigs[c].ancestry[index] for index in ancestors], sites_by_contig[c], recombination_rate,
                      mutation_rate, chromosome_maps[c])
                     for c in range(n_contigs) for first in range(0, len(members), per_task)]
            offspring = [[None] * len(members) for _ in range(n_contigs)]
            for contig_index, first, children in pool.imap_unordered(_simulate_children, tasks):
                offspring[contig_index][first:first + len(children)] = children
            for c in range(n_contigs):
                for position, (haplotypes, ancestry, events) in enumerate(offspring[c]):
                    index = members[position]
                    contigs[c].haplotypes[index] = haplotypes
                    contigs[c].ancestry[index] = ancestry
                    for slot, parent in ((0, rows[index].parent1), (1, rows[index].parent2)):
                        contigs[c].gametes.append((index, slot, parent, events[slot]))
    for contig in contigs:
        contig.gametes.sort(key=lambda gamete: (gamete[0], gamete[1]))
    table = PedigreeTable(tuple(row.sample for row in rows), tuple(depth[row.sample] for row in rows),
                          tuple(row.parent1 for row in rows), tuple(row.parent2 for row in rows))
    return table, contigs
