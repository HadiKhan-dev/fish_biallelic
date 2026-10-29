"""Generating design of a known-pedigree simulation (never inference settings).

Every field is user-visible through ``[simulation]`` / ``[simulation.reads]``. The generating read
model and recombination rate are deliberately independent of the inference read model and map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SimulationDesign:
    """Pedigree structure and observation.

    backcross_fraction: probability that a child of cohort >= backcross_start_cohort is a
    backcross (a member of the previous cohort with one of its actual parents) instead of a cross
    of two previous-cohort members. observe_cohorts: cohorts whose members are sequenced (empty:
    all); observed_fraction: random fraction of those kept (seeded, after the pedigree is made).
    pedigree (release 2.17): an explicit pedigree table (:func:`read_pedigree_table`) that replaces
    the cohort sizes and the random matings; its ``observed`` column chooses who is sequenced
    (founders may be), so the observation options above must keep their defaults.
    """

    backcross_fraction: float = 0.0
    backcross_start_cohort: int = 3
    observe_cohorts: tuple[int, ...] = ()
    observed_fraction: float = 1.0
    pedigree: Path | None = None

    def __post_init__(self):
        object.__setattr__(self, "observe_cohorts", tuple(int(c) for c in self.observe_cohorts))
        if self.pedigree is not None:
            object.__setattr__(self, "pedigree", Path(self.pedigree))
            if self.backcross_fraction or self.observe_cohorts or self.observed_fraction != 1:
                raise ValueError("an explicit pedigree sets matings and observation itself: leave backcross_fraction, "
                                 "observe_cohorts and observed_fraction at their defaults")
        if not 0 <= self.backcross_fraction <= 1:
            raise ValueError("backcross_fraction must be between zero and one")
        if self.backcross_start_cohort < 2:
            raise ValueError("backcrosses need a generated individual and one actual parent")
        if not 0 < self.observed_fraction <= 1:
            raise ValueError("observed_fraction must be in (0, 1]")
        if any(c < 1 for c in self.observe_cohorts) or len(set(self.observe_cohorts)) != len(self.observe_cohorts):
            raise ValueError("observe_cohorts must be unique positive cohort numbers")


@dataclass(frozen=True)
class ReadModel:
    """Generating read model (inference keeps its own calibrated model).

    Per cell, depth ~ Poisson(depth x sample factor) (factor lognormal with coefficient of
    variation depth_cv, mean one), one dropout tract of dropout_fraction of the sites per sample,
    and ALT reads ~ Binomial(depth, p_g) with p = (error_rate, heterozygote_alt_probability,
    1 - error_rate). read_length_bp > 0 draws paired-end fragments instead (insert length
    Normal(fragment_mean_bp, fragment_sd_bp)), so sites closer than a fragment share reads.
    """

    error_rate: float = 0.02
    depth_cv: float = 0.0
    heterozygote_alt_probability: float = 0.5
    dropout_fraction: float = 0.0
    read_length_bp: int = 0
    fragment_mean_bp: float = 375.0
    fragment_sd_bp: float = 150.0

    def __post_init__(self):
        if not 0 < self.error_rate < 0.5:
            raise ValueError("generating error_rate must be between zero and 0.5")
        if not math.isfinite(self.depth_cv) or self.depth_cv < 0:
            raise ValueError("depth_cv must be finite and nonnegative")
        if not 0 < self.heterozygote_alt_probability < 1:
            raise ValueError("heterozygote_alt_probability must be in (0, 1)")
        if not 0 <= self.dropout_fraction < 1:
            raise ValueError("dropout_fraction must be in [0, 1)")
        if int(self.read_length_bp) != self.read_length_bp or self.read_length_bp < 0:
            raise ValueError("read_length_bp must be a nonnegative integer")
        if self.read_length_bp and not (
                math.isfinite(self.fragment_mean_bp) and self.fragment_mean_bp > 0
                and math.isfinite(self.fragment_sd_bp) and self.fragment_sd_bp >= 0):
            raise ValueError("fragment length mean must be positive and its sd nonnegative")


@dataclass(frozen=True)
class SimulationConfig:
    """A known-pedigree simulation from founder templates.

    templates: directory of ``<contig>.npz`` founder templates (``positions`` int64[L],
    ``allele_probabilities`` float[H, L, 2]: H founder haplotypes, consecutive pairs form the
    founders). contigs: simulated contigs, in this order (the order is part of the random
    streams). seed: None = drawn at random once and bound in the store. cohorts: cohort sizes
    (cohort 1 descends from the founders). depth: mean read depth. generating_rate_cm_per_mb /
    generating_map: the generating genetic map (map file contigs; others the rate).
    mutation_rate_per_bp: per-site mutation probability per meiosis (internal constant).
    """

    templates: Path
    contigs: tuple[str, ...]
    seed: int | None = 400
    cohorts: tuple[int, ...] = (20, 100, 200)
    depth: float = 5.0
    generating_rate_cm_per_mb: float = 5.0
    generating_map: Path | None = None
    design: SimulationDesign = field(default_factory=SimulationDesign)
    reads: ReadModel = field(default_factory=ReadModel)
    mutation_rate_per_bp: float = 1e-10

    def __post_init__(self):
        object.__setattr__(self, "templates", Path(self.templates))
        object.__setattr__(self, "contigs", tuple(map(str, self.contigs)))
        object.__setattr__(self, "cohorts", tuple(int(size) for size in self.cohorts))
        if self.generating_map is not None:
            object.__setattr__(self, "generating_map", Path(self.generating_map))
        if not self.contigs or len(set(self.contigs)) != len(self.contigs):
            raise ValueError("simulation contigs must be unique and non-empty")
        if not self.cohorts or any(size < 2 for size in self.cohorts):
            raise ValueError("each generated cohort must have at least two individuals")
        if not math.isfinite(self.depth) or self.depth <= 0:
            raise ValueError("depth must be finite and positive")
        if self.seed is not None and int(self.seed) != self.seed:
            raise ValueError("seed must be an integer or None (random)")


PEDIGREE_COLUMNS = ("sample", "parent1", "parent2", "sex", "generation", "observed")


@dataclass(frozen=True)
class PedigreeRow:
    """One individual of an explicit pedigree: parents empty for a founder (founders take the template founders
    in table order); parent1 gives the first (maternal) haplotype; observed: sequenced."""

    sample: str
    parent1: str
    parent2: str
    sex: str
    generation: str
    observed: bool


def read_pedigree_table(path) -> tuple[PedigreeRow, ...]:
    """The rows of an explicit pedigree table (tab-separated, header PEDIGREE_COLUMNS in any order, more columns
    allowed). Founders (both parents empty) come first; every other row names two different earlier rows as
    parents. Sample names are unique; observed is 0/1 (or false/true)."""
    path = Path(path)
    lines = [line.rstrip("\n").split("\t") for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path}: empty pedigree table")
    header = lines[0]
    missing = [name for name in PEDIGREE_COLUMNS if name not in header]
    if missing:
        raise ValueError(f"{path}: missing pedigree columns {', '.join(missing)}")
    column = {name: header.index(name) for name in PEDIGREE_COLUMNS}
    rows, seen, founders_done = [], set(), False
    for number, fields in enumerate(lines[1:], start=2):
        fields = fields + [""] * (len(header) - len(fields))
        value = {name: fields[index].strip() for name, index in column.items()}
        observed = value["observed"].lower()
        if observed not in ("0", "1", "false", "true"):
            raise ValueError(f"{path}:{number}: observed must be 0/1 or false/true")
        row = PedigreeRow(value["sample"], value["parent1"], value["parent2"], value["sex"], value["generation"],
                          observed in ("1", "true"))
        if not row.sample or row.sample in seen:
            raise ValueError(f"{path}:{number}: sample names must be unique and non-empty")
        founder = not row.parent1 and not row.parent2
        if founder and founders_done:
            raise ValueError(f"{path}:{number}: founders must precede every other row")
        if not founder:
            founders_done = True
            if not row.parent1 or not row.parent2 or row.parent1 == row.parent2:
                raise ValueError(f"{path}:{number}: a non-founder needs two different parents")
            unknown = [parent for parent in (row.parent1, row.parent2) if parent not in seen]
            if unknown:
                raise ValueError(f"{path}:{number}: parents must be earlier rows ({', '.join(unknown)})")
        seen.add(row.sample)
        rows.append(row)
    if not any(row.observed for row in rows):
        raise ValueError(f"{path}: no observed individual")
    return tuple(rows)


def observed_indices(cohort_of_rows, design: SimulationDesign, seed: int) -> np.ndarray:
    """Rows (pedigree order) of the sequenced individuals, chosen after the whole pedigree exists."""
    cohort = np.asarray(cohort_of_rows, dtype=np.int64)
    if design.observe_cohorts and set(design.observe_cohorts) - set(cohort.tolist()):
        raise ValueError("an observed cohort was not generated")
    keep = np.arange(len(cohort))
    if design.observe_cohorts:
        keep = keep[np.isin(cohort, design.observe_cohorts)]
    if design.observed_fraction < 1:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 3_000_000]))
        keep = np.sort(rng.choice(keep, size=int(len(keep) * design.observed_fraction), replace=False))
    if len(keep) < 3:
        raise ValueError("observation selection retains fewer than three individuals")
    return keep
