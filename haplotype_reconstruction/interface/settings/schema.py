"""The run configuration: one typed tree of user settings (TOML sections), strictly validated.

Sections: run (paths, final-analysis contigs, where to stop), execution (CPUs,
lanes, scratch; never scientific identity), input (VCF or simulation, contigs),
quality (QC and read calibration), genetic_map (inference prior),
family_phase, recombination, export, and simulation (source, genome, sites, stocks,
reads and artefacts; generating parameters only). Internal constants remain in
stage parameter modules; stages/configure.py maps settings into them.

The pipeline reads only the observations (VCF/BCF, or the simulated reads) and the genetic map. Sample names
are identifiers. The settings loader rejects sample-metadata sections.

Every field carries its kind (how TOML values are parsed and checked) and a one-line description (docs are
generated from it). Paths are resolved against the working directory when the configuration is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ...simulation import parameters as generating


def setting(default: Any, kind: str, doc: str, choices: tuple = (), *, affects_results: bool = True) -> Any:
    """A configuration field: ``kind`` is one of bool, int, float, str, path, strings, ints, cores, seed, memory."""
    if isinstance(default, (dict, list)):
        raise TypeError("mutable defaults are not allowed")
    return field(
        default=default,
        metadata=dict(
            kind=kind,
            doc=doc,
            choices=tuple(choices),
            affects_results=affects_results,
            provenance=doc,
        ),
    )


def section(factory: Any, doc: str) -> Any:
    return field(
        default_factory=factory,
        metadata=dict(
            kind="section",
            doc=doc,
            choices=(),
            affects_results=any(item.metadata["affects_results"] for item in fields(factory)),
        ),
    )


@dataclass(frozen=True)
class RunSection:
    output: Path | None = setting(
        None,
        "path",
        "Run directory (outputs, logs, run records); default work/runs/<preset or input kind>[_seed<seed>].",
        affects_results=False,
    )
    checkpoints: Path | None = setting(
        None, "path", "Checkpoint directory; default OUTPUT/checkpoints.", affects_results=False
    )
    exclude_contigs: tuple[str, ...] = setting(
        (),
        "strings",
        "Configured contigs to omit from pedigree and final downstream analysis. "
        "Evidence and relatedness still use every input contig; run skips excluded reconstruction "
        "and retains its existing records. Change this set on resume to compute separate final products.",
    )
    stop_after: str = setting(
        "",
        "str",
        "Stop cleanly after this stage: relatedness (every contig's evidence and "
        "the genome-wide relatedness product recorded; run it once before shards), discovery, "
        "or painting (per-contig work done, later genome-wide stages not started); empty runs "
        "everything.",
        ("", "relatedness", "discovery", "painting"),
        affects_results=False,
    )


@dataclass(frozen=True)
class ExecutionSection:
    cores: int | str = setting(
        "auto", "cores", "Total CPU ceiling (processes x threads); auto = CPU affinity.", affects_results=False
    )
    contig_lanes: int | str = setting(
        "auto",
        "cores",
        "Contigs reconstructed concurrently; auto = one lane per 16 "
        "cores, paced by measured memory pressure; 1 = one contig at a time with all cores.",
        affects_results=False,
    )
    memory_limit: int | str = setting(
        "auto",
        "memory",
        "Capacity for measured job memory pressure: a size such as "
        '"200G" (binary units), or auto = job cgroup capacity. SOFT/HARD '
        "thresholds and sampling interval are in engine/memory.py; affects scheduling only.",
        affects_results=False,
    )
    serial: bool = setting(
        False,
        "bool",
        "Discover, assemble and paint one contig at a time, each recorded before the next (short allocations).",
        affects_results=False,
    )
    max_contigs: int = setting(
        0,
        "int",
        "Finish at most this many more contigs in this invocation, then stop "
        "cleanly (0 = no limit; > 0 implies serial).",
        affects_results=False,
    )
    profile: bool = setting(
        False, "bool", "Print per-phase timings and thread-allocation traces.", affects_results=False
    )

    scratch: Path | None = setting(
        None,
        "path",
        "Mapped scratch and kernel-cache base; defaults to the checkpoint filesystem.",
        affects_results=False,
    )


@dataclass(frozen=True)
class InputSection:
    kind: str = setting(
        "vcf",
        "str",
        "Observations: an indexed VCF/BCF with FORMAT/AD, or a simulation from a file panel or a coalescent genealogy.",
        ("vcf", "simulation"),
    )
    vcf: Path | None = setting(None, "path", "Input VCF/BCF (kind = vcf).")
    contigs: tuple[str, ...] = setting(
        (), "strings", "Contigs to analyse, in order (required unless the preset lists them)."
    )


@dataclass(frozen=True)
class QualitySection:
    site_filter: str = setting(
        "auto",
        "str",
        "Self-calibrated per-contig site filter; auto = on for VCF input, off for simulations.",
        ("auto", "on", "off"),
    )
    sample_filter: str = setting(
        "auto",
        "str",
        "Per-contig library-contamination QC (flagged samples leave "
        "founder construction); auto = on for VCF input, off for simulations; it runs "
        "only where the site filter is applied.",
        ("auto", "on", "off"),
    )
    read_calibration: bool = setting(
        True,
        "bool",
        "Fit read error, allelic balance and overdispersion from the "
        "observed AD; off = fixed binomial model (error 0.02).",
    )


@dataclass(frozen=True)
class GeneticMapSection:
    rate_cm_per_mb: float = setting(
        5.0, "float", "Fallback recombination rate (cM/Mb) without a map or beyond its knots."
    )
    file: Path | None = setting(None, "path", "Cumulative cM map (PLINK, or chromosome/position_bp/map_cm TSV).")
    infer_rate: bool = setting(
        False,
        "bool",
        "Use the genome-wide recombination rate estimated from full-sib families in the relatedness "
        "stage as the fallback rate of contigs without a map file, when its gate passes, "
        "rate_cm_per_mb lies outside its 95 % interval and at least five informative families were "
        "used. Off (keep the supplied rate): the estimate is reported, with a warning when "
        "rate_cm_per_mb lies outside the interval, and nothing downstream changes. Map files are "
        "always kept.",
    )


@dataclass(frozen=True)
class FamilyPhaseSection:
    protected_intervals: Path | None = setting(
        None,
        "path",
        "TSV of intervals (chrom, start, end, reason) across "
        "which family phase's meiosis chains reset (reference-order "
        "breakpoints; a reversed_segment row resets at both ends).",
    )


@dataclass(frozen=True)
class RecombinationSection:
    order_breakpoints: Path | None = setting(
        None, "path", "TSV of reference-order breakpoints (chrom, start, end, reason)."
    )


@dataclass(frozen=True)
class ExportSection:
    enabled: bool = setting(
        True,
        "bool",
        "Write the final products at the end of a complete run (OUTPUT/export): per "
        "contig a founder VCF (haploid GT, one column c<component>_H<row> per founder row) and "
        "the phased sample VCF (GT:PS), bgzipped with tabix indexes.",
        affects_results=False,
    )
    full_length: bool = setting(
        True,
        "bool",
        "Write every input biallelic SNP that reached the site filter (FILTER "
        "PASS for kept sites, the rule codes r1-r5 for removed ones); false writes only "
        "the kept sites.",
    )


@dataclass(frozen=True)
class SimulationReadsSection:
    preset: str = setting(
        "", "str", "Generating read preset; explicit read keys override its values.", ("", "realistic", "short_read_cohort")
    )
    error_rate: float = setting(0.02, "float", "Generating sequencing error rate.")
    depth_cv: float = setting(0.0, "float", "Between-sample depth coefficient of variation.")
    heterozygote_alt_probability: float = setting(0.5, "float", "Alternate-read probability in heterozygotes.")
    dropout_fraction: float = setting(0.0, "float", "Contiguous missing-marker fraction per sample.")
    read_length_bp: int = setting(
        0, "int", "Read length; > 0 samples paired-end fragments shared by neighbouring sites."
    )
    fragment_mean_bp: float = setting(375.0, "float", "Mean fragment length (read_length_bp > 0).")
    fragment_sd_bp: float = setting(150.0, "float", "Fragment length standard deviation.")
    site_depth_cv: float = setting(0.0, "float", "Shared site-depth lognormal CV (population mean one); 0 = off.")
    site_depth_window_bp: int = setting(
        0, "int", "Share site factors within physical windows of this width; 0 = independent sites."
    )
    count_dispersion: float = setting(
        0.0, "float", "Gamma-Poisson cell dispersion alpha: variance = mean + alpha * mean squared; 0 = Poisson."
    )
    site_loss_fraction: float = setting(
        generating.SITE_LOSS_FRACTION, "float",
        "Impaired share of site-depth windows; uniform retained share, not renormalised. Depth is unimpaired depth."
    )
    region_depth_cv: float = setting(
        generating.REGION_DEPTH_CV, "float", "Mean-one lognormal depth CV per sample and physical window; 0 = off."
    )
    region_depth_window_bp: int = setting(
        generating.REGION_DEPTH_WINDOW_BP, "int", "Regional depth window width in bp; positive when regional CV is nonzero."
    )
    site_loss_fine_fraction: float = setting(
        generating.SITE_LOSS_FINE_FRACTION, "float",
        "Impaired share of independent fine windows; uniform retained share, multiplied without renormalisation; 0 = off."
    )
    site_loss_fine_window_bp: int = setting(
        generating.SITE_LOSS_FINE_WINDOW_BP, "int", "Fine-loss window width in bp; positive when fine loss is nonzero."
    )



@dataclass(frozen=True)
class SimulationGenomeSection:
    contigs: tuple[str, ...] = setting((), "strings", "Generated contig names; must match input.contigs in order.")
    length_bp: int | tuple = setting(0, "lengths", "Positive integer bp length, or a table keyed by contig.")


@dataclass(frozen=True)
class SimulationSitesSection:
    count: int | None = setting(None, "optional_int", "Exact positive number of distinct written sites, including spurious sites.")
    per_mb: float | None = setting(None, "optional_float", "Expected sites per Mb; exclusive with count and positions.")
    positions: Path | None = setting(None, "path", "Sorted unique integer bp positions file; profile must be none.")
    profile: str = setting("none", "str", "Spatial intensity profile.", ("none", "cascade"))
    sigmas: tuple[float, ...] = setting(
        generating.PROFILE_SIGMAS, "floats", "Log-intensity SDs at 10, 100, 1000, 10000, 100000 bp."
    )
    masked_fraction: float = setting(
        generating.PROFILE_MASKED_FRACTION, "float", "Stationary masked share of the cascade profile."
    )
    masked_mean_bp: float = setting(
        generating.PROFILE_MASKED_MEAN_BP, "float", "Mean exponential masked-stretch length in bp."
    )
    variant_against: str = setting(
        generating.VARIANT_AGAINST, "str", "Variant separating set: carried sources, or carried sources plus reference.",
        ("cohort", "reference")
    )
    spurious_fraction: float = setting(
        generating.SPURIOUS_FRACTION, "float", "Fraction of written sites with reference alleles only; count includes these sites."
    )


@dataclass(frozen=True)
class SimulationPopulationSection:
    size: float = setting(0.0, "float", "Positive diploid effective population size in individuals.")
    parent: str = setting("", "str", "Parent population; empty identifies the single root.")
    split_generations: float | None = setting(
        None, "optional_float", "Finite nonnegative split time in generations; non-root only."
    )


@dataclass(frozen=True)
class SimulationSourceSection:
    kind: str = setting("file", "str", "Source genealogy or a fixed file panel.", ("file", "coalescent"))
    file: Path | None = setting(None, "path", "Directory containing one source NPZ per contig.")
    unknown: str = setting("refuse", "str", "Policy for file alleles coded -1.", ("refuse", "draw"))
    reference_population: str = setting(
        "", "str", "Population of the reference leaf; empty selects the first population."
    )
    populations: tuple = setting((), "populations", "Named constant-size populations and their split tree.")


@dataclass(frozen=True)
class SimulationStockSection:
    population: str = setting("", "str", "Source population of this stock's generation zero.")
    founders: int = setting(0, "int", "Generation-zero diploid individuals; at least two.")
    size: int = setting(0, "int", "Diploid individuals in each later generation; at least two.")
    generations: int = setting(0, "int", "Nonnegative number of random-mating stock generations.")


@dataclass(frozen=True)
class SimulationArtefactsSection:
    preset: str = setting("", "str", "Named call-set preset; explicit keys override it.", ("", "reference_blocks"))
    enabled: bool = setting(False, "bool", "Apply call-set artefacts after read sampling.")
    hard_sites: float = setting(generating.ARTEFACT_HARD_SITES, "float", "Probability of a hard site.")
    hard_miss: float = setting(generating.ARTEFACT_HARD_MISS, "float", "Miss probability contributed by a hard site.")
    stretch_cover: float = setting(
        generating.ARTEFACT_STRETCH_COVER, "float", "Long-axis low-sensitivity covered share."
    )
    stretch_sites: float = setting(
        generating.ARTEFACT_STRETCH_SITES, "float", "Mean geometric stretch length in sites."
    )
    stretch_miss: float = setting(
        generating.ARTEFACT_STRETCH_MISS, "float", "Miss probability inside a low-sensitivity stretch."
    )
    miss: float = setting(generating.ARTEFACT_MISS, "float", "Baseline non-reference miss probability.")
    miss_sample_cv: float = setting(
        generating.ARTEFACT_MISS_SAMPLE_CV, "float", "Sample miss-factor coefficient of variation."
    )
    call_min_alt_fraction: float = setting(
        generating.ARTEFACT_CALL_MIN_ALT_FRACTION, "float", "Minimum ALT fraction calling a hom-reference cell."
    )
    block_band: float = setting(generating.ARTEFACT_BLOCK_BAND, "float", "Relative logarithmic depth-band width.")
    balance_dispersion: float = setting(
        generating.ARTEFACT_BALANCE_DISPERSION, "float", "Beta-binomial heterozygote intraclass correlation."
    )
    systematic_sites: float = setting(
        generating.ARTEFACT_SYSTEMATIC_SITES, "float", "Probability of a systematic ALT-error site."
    )
    systematic_lo: float = setting(
        generating.ARTEFACT_SYSTEMATIC_LO, "float", "Lower systematic ALT-error probability."
    )
    systematic_hi: float = setting(
        generating.ARTEFACT_SYSTEMATIC_HI, "float", "Upper systematic ALT-error probability."
    )
    regions: tuple = setting((), "regions", "Intervals {contig, start, end, factor}; half-open bp bounds, no overlaps.")


NAMED_SECTIONS = {"populations": SimulationPopulationSection, "stocks": SimulationStockSection}


@dataclass(frozen=True)
class SimulationSection:
    seed: int | str = setting(400, "seed", 'Simulation seed, or "random" (the realised seed is recorded).')
    founders: tuple = setting((), "counts", "Random cross founders by stock: {STOCK = number of individuals}.")
    genome: SimulationGenomeSection = section(SimulationGenomeSection, "Generated genome extent.")
    sites: SimulationSitesSection = section(SimulationSitesSection, "Segregating-site clocks and intensity.")
    source: SimulationSourceSection = section(SimulationSourceSection, "Source panel or structured genealogy.")
    stocks: tuple = setting((), "stocks", "Named stock pedigrees preceding the cross.")
    artefacts: SimulationArtefactsSection = section(
        SimulationArtefactsSection, "Call-set artefact model, off by default."
    )
    cohorts: tuple[int, ...] = setting((20, 100, 200), "ints", "Individuals in successive cohorts.")
    depth: float = setting(5.0, "float", "Unimpaired site depth; mean is depth * (1 - site_loss_fraction / 2) * (1 - site_loss_fine_fraction / 2).")
    generating_rate_cm_per_mb: float = setting(5.0, "float", "Generating fallback recombination rate.")
    generating_map: Path | None = setting(None, "path", "Generating genetic map (independent of inference).")
    backcross_fraction: float = setting(0.0, "float", "Fraction of later matings with an actual parent.")
    backcross_start_cohort: int = setting(3, "int", "First cohort with backcrosses.")
    observe_cohorts: tuple[int, ...] = setting((), "ints", "Only sequence these cohorts (empty: all).")
    observed_fraction: float = setting(1.0, "float", "Random fraction of eligible individuals sequenced.")
    pedigree: Path | None = setting(
        None,
        "path",
        "Explicit pedigree table (TSV: sample, parent1, parent2, sex, "
        "generation, observed, stock); replaces cohorts and the random matings.",
    )
    reads: SimulationReadsSection = section(SimulationReadsSection, "Generating read model.")


@dataclass(frozen=True)
class Settings:
    """The resolved configuration of one run (see the module docstring)."""

    version: int = setting(4, "int", "Settings schema version (4).", affects_results=False)
    preset: str = setting("", "str", "Built-in base configuration: tromau or generic.", ("", "tromau", "generic"))
    run: RunSection = section(RunSection, "Output paths, final-analysis contigs and where to stop.")
    execution: ExecutionSection = section(ExecutionSection, "CPUs, lanes and scratch (never part of results).")
    input: InputSection = section(InputSection, "Observations and contigs.")
    quality: QualitySection = section(QualitySection, "Evidence quality control and read model.")
    genetic_map: GeneticMapSection = section(GeneticMapSection, "Inference genetic map.")
    family_phase: FamilyPhaseSection = section(FamilyPhaseSection, "Family phase.")
    recombination: RecombinationSection = section(RecombinationSection, "Recombination maps.")
    export: ExportSection = section(ExportSection, "Final products: founder and sample VCFs.")
    simulation: SimulationSection = section(SimulationSection, "Simulation (input kind = simulation).")

    @property
    def serial(self) -> bool:
        """Contigs one at a time ([execution] serial, implied by max_contigs > 0)."""
        return self.execution.serial or self.execution.max_contigs > 0


def schema_fields(cls) -> list:
    """The configuration fields of a section class, in declaration order."""
    return [item for item in fields(cls) if "kind" in item.metadata]


def to_dict(value) -> Any:
    """Plain data (paths as strings, None as "") of a config object."""
    if hasattr(value, "__dataclass_fields__"):
        return {
            item.name: (
                {name: to_dict(v) for name, v in getattr(value, item.name)}
                if item.metadata["kind"] in (*NAMED_SECTIONS, "counts")
                or (item.metadata["kind"] == "lengths" and isinstance(getattr(value, item.name), tuple))
                else to_dict(getattr(value, item.name))
            )
            for item in schema_fields(type(value))
        }
    if isinstance(value, tuple):
        return [to_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return ""
    return value


def reference_rows(cls=Settings, prefix: tuple = ()) -> list[dict]:
    """Every configuration key with its section, kind, default, choices and description (docs/configuration.md is
    generated from these rows, so the documentation cannot drift from the schema)."""
    rows = []
    for item in schema_fields(cls):
        kind = item.metadata["kind"]
        if kind == "section":
            rows.extend(reference_rows(type(item.default_factory()), (*prefix, item.name)))
            continue
        if kind in NAMED_SECTIONS:
            rows.extend(reference_rows(NAMED_SECTIONS[kind], (*prefix, item.name, "NAME")))
            continue
        default = item.default
        rows.append(
            dict(
                section=".".join(prefix),
                key=item.name,
                kind=kind,
                default=to_dict(default),
                choices=[choice for choice in item.metadata["choices"] if choice != ""],
                doc=item.metadata["doc"],
                affects_results=item.metadata["affects_results"],
                provenance=item.metadata["provenance"],
            )
        )
    return rows


__all__ = [
    "Settings",
    "RunSection",
    "ExecutionSection",
    "InputSection",
    "QualitySection",
    "GeneticMapSection",
    "FamilyPhaseSection",
    "RecombinationSection",
    "ExportSection",
    "SimulationReadsSection",
    "SimulationSection",
    "schema_fields",
    "to_dict",
    "reference_rows",
]
