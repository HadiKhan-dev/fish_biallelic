"""Generating design of a known-pedigree simulation (never inference settings).

Every field is user-visible through ``[simulation]`` / ``[simulation.reads]``. The generating read
model and recombination rate are deliberately independent of the inference read model and map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..contracts.roles import record_field
from ..core.errors import InputError, SettingsError
from ..core.random import stream
from . import parameters as generating
from .artefacts import ArtefactModel
from .artefacts import ArtefactRegion as ArtefactRegion
from .parameters import MUTATION_RATE_PER_BP, PROFILE_MASKED_FRACTION, PROFILE_MASKED_MEAN_BP, PROFILE_SIGMAS
from .sites import Profile, SiteMode


@dataclass(frozen=True)
class SimulationDesign:
    """Pedigree structure and observation.

    backcross_fraction: probability that a child of cohort >= backcross_start_cohort is a
    backcross (a member of the previous cohort with one of its actual parents) instead of a cross
    of two previous-cohort members. observe_cohorts: cohorts whose members are sequenced (empty:
    all); observed_fraction: random fraction of those kept (seeded, after the pedigree is made).
    pedigree: an explicit pedigree table (:func:`read_pedigree_table`) that replaces
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
                raise SettingsError(
                    "an explicit pedigree sets matings and observation itself: leave backcross_fraction, "
                    "observe_cohorts and observed_fraction at their defaults"
                )
        if not 0 <= self.backcross_fraction <= 1:
            raise SettingsError("backcross_fraction must be between zero and one")
        if self.backcross_start_cohort < 2:
            raise SettingsError("backcrosses need a generated individual and one actual parent")
        if not 0 < self.observed_fraction <= 1:
            raise SettingsError("observed_fraction must be in (0, 1]")
        if any(c < 1 for c in self.observe_cohorts) or len(set(self.observe_cohorts)) != len(self.observe_cohorts):
            raise SettingsError("observe_cohorts must be unique positive cohort numbers")


@dataclass(frozen=True)
class ReadModel:
    """Generating read model (inference keeps its own calibrated model).

    Per cell, depth ~ Poisson(depth x sample factor) (factor lognormal with coefficient of
    variation depth_cv, mean one), one dropout tract of dropout_fraction of the sites per sample,
    and ALT reads ~ Binomial(depth, p_g) with p = (error_rate, heterozygote_alt_probability,
    1 - error_rate). read_length_bp > 0 draws paired-end fragments instead (insert length
    Normal(fragment_mean_bp, fragment_sd_bp)), so sites closer than a fragment share reads.
    site_depth_cv adds a mean-one lognormal site factor shared by samples; a positive
    site_depth_window_bp shares that factor within fixed physical windows instead. Independent
    mean-one gamma cell factors give count variance mu + count_dispersion * mu**2 conditional
    on sample/site factors. site_loss_fraction impairs a shared site window with that
    probability and retains a uniform share of reads, without renormalisation: depth is
    the depth at an unimpaired site and the mean is depth * (1 - site_loss_fraction / 2)
    * (1 - site_loss_fine_fraction / 2). The independent fine loss uses occupied windows
    of site_loss_fine_window_bp, multiplies the coarse factor, and is drawn after it.
    region_depth_cv supplies an independent mean-one lognormal factor per sample and
    physical window of region_depth_window_bp. Zero CV/dispersion consumes no extra random draws.
    """

    error_rate: float = 0.02
    depth_cv: float = 0.0
    heterozygote_alt_probability: float = 0.5
    dropout_fraction: float = 0.0
    read_length_bp: int = 0
    fragment_mean_bp: float = 375.0
    fragment_sd_bp: float = 150.0
    site_depth_cv: float = 0.0
    site_depth_window_bp: int = 0
    count_dispersion: float = 0.0
    site_loss_fraction: float = record_field("obs", units="1", dtype="float64", default=generating.SITE_LOSS_FRACTION)
    region_depth_cv: float = record_field("obs", units="1", dtype="float64", default=generating.REGION_DEPTH_CV)
    region_depth_window_bp: int = record_field("obs", units="bp", dtype="int", default=generating.REGION_DEPTH_WINDOW_BP)

    site_loss_fine_fraction: float = record_field(
        "obs", units="1", dtype="float64", default=generating.SITE_LOSS_FINE_FRACTION
    )
    site_loss_fine_window_bp: int = record_field(
        "obs", units="bp", dtype="int", default=generating.SITE_LOSS_FINE_WINDOW_BP
    )

    def __post_init__(self):
        if not 0 < self.error_rate < 0.5:
            raise SettingsError("generating error_rate must be between zero and 0.5")
        if not math.isfinite(self.depth_cv) or self.depth_cv < 0:
            raise SettingsError("depth_cv must be finite and nonnegative")
        if not math.isfinite(self.site_depth_cv) or self.site_depth_cv < 0:
            raise SettingsError("site_depth_cv must be finite and nonnegative")
        if not math.isfinite(self.site_depth_window_bp) or (
            int(self.site_depth_window_bp) != self.site_depth_window_bp or self.site_depth_window_bp < 0
        ):
            raise SettingsError("site_depth_window_bp must be a nonnegative integer")
        if not 0 <= self.site_loss_fraction < 1:
            raise SettingsError("site_loss_fraction must be in [0, 1)")
        if not 0 <= self.site_loss_fine_fraction < 1:
            raise SettingsError("site_loss_fine_fraction must be in [0, 1)")
        if not math.isfinite(self.site_loss_fine_window_bp) or (
            int(self.site_loss_fine_window_bp) != self.site_loss_fine_window_bp or self.site_loss_fine_window_bp < 0
        ):
            raise SettingsError("site_loss_fine_window_bp must be a nonnegative integer")
        if self.site_loss_fine_fraction and not self.site_loss_fine_window_bp:
            raise SettingsError("positive site_loss_fine_fraction requires positive site_loss_fine_window_bp")
        if not math.isfinite(self.region_depth_cv) or self.region_depth_cv < 0:
            raise SettingsError("region_depth_cv must be finite and nonnegative")
        if not math.isfinite(self.region_depth_window_bp) or (
            int(self.region_depth_window_bp) != self.region_depth_window_bp or self.region_depth_window_bp < 0
        ):
            raise SettingsError("region_depth_window_bp must be a nonnegative integer")
        if self.region_depth_cv and not self.region_depth_window_bp:
            raise SettingsError("positive region_depth_cv requires positive region_depth_window_bp")
        if not math.isfinite(self.count_dispersion) or self.count_dispersion < 0:
            raise SettingsError("count_dispersion must be finite and nonnegative")
        if not 0 < self.heterozygote_alt_probability < 1:
            raise SettingsError("heterozygote_alt_probability must be in (0, 1)")
        if not 0 <= self.dropout_fraction < 1:
            raise SettingsError("dropout_fraction must be in [0, 1)")
        if int(self.read_length_bp) != self.read_length_bp or self.read_length_bp < 0:
            raise SettingsError("read_length_bp must be a nonnegative integer")
        if self.read_length_bp and not (
            math.isfinite(self.fragment_mean_bp)
            and self.fragment_mean_bp > 0
            and math.isfinite(self.fragment_sd_bp)
            and self.fragment_sd_bp >= 0
        ):
            raise SettingsError("fragment length mean must be positive and its sd nonnegative")


@dataclass(frozen=True)
class PopulationSpec:
    """A constant diploid effective size and an optional split into its parent."""

    name: str = record_field("obs", dtype="str")
    size: float = record_field("obs", units="individuals", dtype="float64")
    parent: str = record_field("obs", dtype="str", default="")
    split_generations: float | None = record_field("obs", units="generations", dtype="float64 | None", default=None)


@dataclass(frozen=True)
class StockSpec:
    """Generation zero founders, then size individuals for each of generations rounds."""

    name: str = record_field("obs", dtype="str")
    population: str = record_field("obs", dtype="str")
    founders: int = record_field("obs", units="individuals", dtype="int")
    size: int = record_field("obs", units="individuals", dtype="int")
    generations: int = record_field("obs", units="generations", dtype="int")

    def __post_init__(self):
        for name, lower in (("founders", 2), ("size", 2), ("generations", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise SettingsError(f"stock {self.name}: {name} must be an integer >= {lower}")


@dataclass(frozen=True)
class SourceSpec:
    """File coding or a structured coalescent with one reference leaf."""

    kind: str = record_field("obs", dtype="str", default="file")
    file: Path | None = record_field("obs", dtype="Path | None", default=None)
    unknown: str = record_field("obs", dtype="str", default="refuse")
    reference_population: str = record_field("obs", dtype="str", default="")
    populations: tuple[PopulationSpec, ...] = record_field("obs", dtype="tuple[PopulationSpec, ...]", default=())

    def __post_init__(self):
        if self.kind not in ("file", "coalescent"):
            raise SettingsError("source kind must be file or coalescent")
        if self.unknown not in ("refuse", "draw"):
            raise SettingsError("source unknown must be refuse or draw")
        if self.file is not None:
            object.__setattr__(self, "file", Path(self.file))
        if self.kind == "file":
            if self.file is None:
                raise SettingsError("file source needs simulation.source.file")
            if self.populations or self.reference_population:
                raise SettingsError("file source cannot configure coalescent populations")
        else:
            if self.file is not None or self.unknown != "refuse":
                raise SettingsError("coalescent source cannot configure file or unknown")
            self.population_tree()

    def population_tree(self):
        """Validate the named tree before draws and return its kernel arrays."""
        from .genealogy import validate_population_tree

        names = tuple(p.name for p in self.populations)
        if not names or len(set(names)) != len(names) or any(not n for n in names):
            raise SettingsError("coalescent populations must have unique nonempty names")
        if self.reference_population and self.reference_population not in names:
            raise SettingsError("unknown source reference_population")
        if any(p.parent and p.parent not in names for p in self.populations):
            raise SettingsError("unknown source population parent")
        if any(p.parent and p.split_generations is None for p in self.populations):
            raise SettingsError("each non-root population needs split_generations")
        if any(not p.parent and p.split_generations is not None for p in self.populations):
            raise SettingsError("the root population has no split_generations")
        parent = np.array([names.index(p.parent) if p.parent else -1 for p in self.populations], np.int64)
        split = np.array([p.split_generations if p.parent else np.inf for p in self.populations], float)
        size = np.array([p.size for p in self.populations], float)
        validate_population_tree(np.array([0, 0], np.int64), parent, split, size)
        return names, parent, split, size


@dataclass(frozen=True)
class ProfileSpec:
    """JSON-compatible profile specification; arrays are made only for generation."""

    kind: str = record_field("obs", dtype="str", default="none")
    sigmas: tuple[float, ...] = record_field("obs", axes=("scale",), dtype="tuple[float, ...]", default=PROFILE_SIGMAS)
    masked_fraction: float = record_field("obs", dtype="float64", default=PROFILE_MASKED_FRACTION)
    masked_mean_bp: float = record_field("obs", units="bp", dtype="float64", default=PROFILE_MASKED_MEAN_BP)

    def __post_init__(self):
        object.__setattr__(self, "sigmas", tuple(self.sigmas))
        self.parameters()

    def parameters(self):
        return Profile(self.kind, self.sigmas, self.masked_fraction, self.masked_mean_bp)


@dataclass(frozen=True)
class SiteSpec:
    """Exactly one site-selection mode and a spatial intensity profile."""

    count: int | None = record_field("obs", units="sites", dtype="int | None", default=None)
    per_mb: float | None = record_field("obs", units="sites/Mb", dtype="float64 | None", default=None)
    positions: Path | None = record_field("obs", dtype="Path | None", default=None)
    profile: ProfileSpec = record_field("obs", dtype="ProfileSpec", default_factory=ProfileSpec)

    variant_against: str = record_field("obs", dtype="str", default=generating.VARIANT_AGAINST)
    spurious_fraction: float = record_field("obs", units="1", dtype="float64", default=generating.SPURIOUS_FRACTION)

    def __post_init__(self):
        if self.variant_against not in ("cohort", "reference"):
            raise SettingsError("variant_against must be cohort or reference")
        if not 0 <= self.spurious_fraction < 1:
            raise SettingsError("spurious_fraction must be in [0, 1)")

    def mode(self, length):
        if self.positions is None:
            mode = SiteMode(count=self.count, per_mb=self.per_mb)
        else:
            if self.count is not None or self.per_mb is not None or self.profile.kind != "none":
                raise SettingsError("positions requires profile = none and no count or per_mb")
            try:
                text = Path(self.positions).read_text().split()
                values = np.array([int(v) for v in text], np.int64)
            except (OSError, ValueError, OverflowError) as error:
                raise SettingsError(f"cannot read integer site positions: {error}") from error
            mode = SiteMode(positions=values)
            if np.any(values < 1) or np.any(values > length):
                raise SettingsError("site positions must lie in the generated genome extent")
        if mode.count is not None and mode.count > length:
            raise SettingsError("site count exceeds the number of integer genome positions")
        if mode.per_mb is not None and mode.per_mb * length / 1e6 >= length:
            raise SettingsError("per_mb target must be below the number of integer genome positions")
        return mode


@dataclass(frozen=True)
class SimulationConfig:
    """A known-pedigree simulation from founder templates.

    templates: directory of ``<contig>.npz`` founder templates (``positions`` int64[L],
    ``allele_probabilities`` float[H, L, 2]: H founder haplotypes, consecutive pairs form the
    founders). contigs: simulated contigs, in this order (the order is part of the random
    streams). seed: None = drawn at random once and bound in the store. cohorts: cohort sizes
    (cohort 1 descends from the founders). depth: read depth at an unimpaired site; site loss reduces the mean by (1 - site_loss_fraction / 2) * (1 - site_loss_fine_fraction / 2).
    generating_rate_cm_per_mb /
    generating_map: the generating genetic map (map file contigs; others the rate).
    mutation_rate_per_bp: per-site mutation probability per meiosis (internal constant).
    """

    templates: Path | None = None
    contigs: tuple[str, ...] = ()
    seed: int | None = 400
    cohorts: tuple[int, ...] = (20, 100, 200)
    depth: float = 5.0
    generating_rate_cm_per_mb: float = 5.0
    generating_map: Path | None = None
    design: SimulationDesign = field(default_factory=SimulationDesign)
    reads: ReadModel = field(default_factory=ReadModel)
    mutation_rate_per_bp: float = MUTATION_RATE_PER_BP
    source: SourceSpec | None = None
    stocks: tuple[StockSpec, ...] = ()
    founders: tuple[tuple[str, int], ...] = ()
    genome_contigs: tuple[str, ...] = ()
    length_bp: int | tuple[tuple[str, int], ...] = 0
    sites: SiteSpec | None = None
    artefacts: ArtefactModel = field(default_factory=ArtefactModel)

    def __post_init__(self):
        if self.templates is not None:
            object.__setattr__(self, "templates", Path(self.templates))
        if self.source is None:
            object.__setattr__(self, "source", SourceSpec(file=self.templates))
        elif self.templates is not None and (self.source.kind != "file" or self.source.file != self.templates):
            raise SettingsError("templates and source.file disagree")
        object.__setattr__(self, "templates", self.source.file)
        object.__setattr__(self, "contigs", tuple(map(str, self.contigs)))
        object.__setattr__(self, "cohorts", tuple(int(size) for size in self.cohorts))
        if self.generating_map is not None:
            object.__setattr__(self, "generating_map", Path(self.generating_map))
        if not self.contigs or len(set(self.contigs)) != len(self.contigs):
            raise SettingsError("simulation contigs must be unique and non-empty")
        if self.design.pedigree is None and (not self.cohorts or any(size < 2 for size in self.cohorts)):
            raise SettingsError("each generated cohort must have at least two individuals")
        if not math.isfinite(self.depth) or self.depth <= 0:
            raise SettingsError("depth must be finite and positive")
        if self.seed is not None and int(self.seed) != self.seed:
            raise SettingsError("seed must be an integer or None (random)")
        if not math.isfinite(self.generating_rate_cm_per_mb) or self.generating_rate_cm_per_mb < 0:
            raise SettingsError("generating_rate_cm_per_mb must be finite and nonnegative")
        if self.genome_contigs and self.genome_contigs != self.contigs:
            raise SettingsError("simulation.genome.contigs and input.contigs must have the same order")
        if len({s.name for s in self.stocks}) != len(self.stocks):
            raise SettingsError("stock names must be unique")
        if self.source.kind == "coalescent":
            names, *_ = self.source.population_tree()
            if not self.stocks or any(s.population not in names for s in self.stocks):
                raise SettingsError("coalescent stocks must name known source populations")
            if self.sites is None:
                raise SettingsError("coalescent source requires simulation.sites")
            for contig in self.contigs:
                length = self.contig_length(contig)
                self.sites.mode(length)
        elif self.sites is not None or self.genome_contigs or self.length_bp:
            raise SettingsError("file sources fix their sites and extent; genome and sites settings do not apply")
        if self.design.pedigree is not None:
            # The stage reads and validates the table before realising a seed.
            # Keeping I/O there also permits callers to supply parsed rows.
            pass
        elif self.stocks:
            self.founder_stocks(None)
        elif self.founders:
            raise SettingsError("founders requires configured stocks")
        if self.design.pedigree is None:
            if self.design.observe_cohorts and max(self.design.observe_cohorts) > len(self.cohorts):
                raise SettingsError("an observed cohort was not generated")
            candidates = sum(
                n
                for i, n in enumerate(self.cohorts, 1)
                if not self.design.observe_cohorts or i in self.design.observe_cohorts
            )
            if int(candidates * self.design.observed_fraction) < 3:
                raise SettingsError("observation selection retains fewer than three individuals")

    def contig_length(self, contig):
        length = self.length_bp if isinstance(self.length_bp, int) else dict(self.length_bp).get(contig, 0)
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise SettingsError(f"{contig}: genome length_bp must be a positive integer")
        if not isinstance(self.length_bp, int) and set(dict(self.length_bp)) != set(self.contigs):
            raise SettingsError("genome length_bp must name exactly the configured contigs")
        return length

    def founder_stocks(self, rows):
        """Stock name per cross founder in its received order; validate capacity."""
        names = tuple(s.name for s in self.stocks)
        if rows is not None:
            if self.founders:
                raise SettingsError("explicit pedigree sets its own founder stocks; omit founders")
            founders = [r for r in rows if not r.parent1]
            if not self.stocks:
                if any(r.stock for r in rows):
                    raise SettingsError("pedigree stock requires configured stocks")
                return ()
            order = tuple(r.stock or (names[0] if len(names) == 1 else "") for r in founders)
            if any(r.stock for r in rows if r.parent1):
                raise SettingsError("pedigree stock belongs on founder rows only")
        else:
            requests = dict(self.founders)
            if not requests or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in requests.values()):
                raise SettingsError("random cohorts require founders = {STOCK = positive integer}")
            if set(requests) - set(names):
                raise SettingsError("founders names an unknown stock")
            order = tuple(name for name in names for _ in range(requests.get(name, 0)))
        minimum = 1 if rows is not None else 2
        if len(order) < minimum or any(name not in names for name in order):
            raise SettingsError("cross founders require known stocks (stock column required with multiple stocks)")
        for stock in self.stocks:
            available = stock.size if stock.generations else stock.founders
            if order.count(stock.name) > available:
                raise SettingsError(
                    f"stock {stock.name}: requested {order.count(stock.name)} individuals, available {available}"
                )
        return order


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
    stock: str = ""


def parse_pedigree_table(text: str, *, label: str = "pedigree") -> tuple[PedigreeRow, ...]:
    """The rows of an explicit pedigree table (tab-separated, header PEDIGREE_COLUMNS in any order, more columns
    allowed). Founders (both parents empty) come first; every other row names two different earlier rows as
    parents. Sample names are unique; observed is 0/1 (or false/true)."""
    path = label
    lines = [line.rstrip("\n").split("\t") for line in text.splitlines() if line.strip()]
    if not lines:
        raise InputError(f"{path}: empty pedigree table")
    header = lines[0]
    missing = [name for name in PEDIGREE_COLUMNS if name not in header]
    if missing:
        raise InputError(f"{path}: missing pedigree columns {', '.join(missing)}")
    column = {name: header.index(name) for name in (*PEDIGREE_COLUMNS, "stock") if name in header}
    rows, seen, founders_done = [], set(), False
    for number, fields in enumerate(lines[1:], start=2):
        fields = fields + [""] * (len(header) - len(fields))
        value = {name: fields[index].strip() for name, index in column.items()}
        observed = value["observed"].lower()
        if observed not in ("0", "1", "false", "true"):
            raise InputError(f"{path}:{number}: observed must be 0/1 or false/true")
        row = PedigreeRow(
            value["sample"],
            value["parent1"],
            value["parent2"],
            value["sex"],
            value["generation"],
            observed in ("1", "true"),
            value.get("stock", ""),
        )
        if not row.sample or row.sample in seen:
            raise InputError(f"{path}:{number}: sample names must be unique and non-empty")
        founder = not row.parent1 and not row.parent2
        if founder and founders_done:
            raise InputError(f"{path}:{number}: founders must precede every other row")
        if not founder:
            founders_done = True
            if not row.parent1 or not row.parent2 or row.parent1 == row.parent2:
                raise InputError(f"{path}:{number}: a non-founder needs two different parents")
            unknown = [parent for parent in (row.parent1, row.parent2) if parent not in seen]
            if unknown:
                raise InputError(f"{path}:{number}: parents must be earlier rows ({', '.join(unknown)})")
        seen.add(row.sample)
        rows.append(row)
    if not any(row.observed for row in rows):
        raise InputError(f"{path}: no observed individual")
    return tuple(rows)


def observed_indices(cohort_of_rows, design: SimulationDesign, seed: int) -> np.ndarray:
    """Rows (pedigree order) of the sequenced individuals, chosen after the whole pedigree exists."""
    cohort = np.asarray(cohort_of_rows, dtype=np.int64)
    if design.observe_cohorts and set(design.observe_cohorts) - set(cohort.tolist()):
        raise SettingsError("an observed cohort was not generated")
    keep = np.arange(len(cohort))
    if design.observe_cohorts:
        keep = keep[np.isin(cohort, design.observe_cohorts)]
    if design.observed_fraction < 1:
        rng = stream("simulation.observation", seed=seed)
        keep = np.sort(rng.choice(keep, size=int(len(keep) * design.observed_fraction), replace=False))
    if len(keep) < 3:
        raise SettingsError("observation selection retains fewer than three individuals")
    return keep
