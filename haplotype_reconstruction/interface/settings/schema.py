"""The run configuration: one typed tree of user settings (TOML sections), strictly validated.

Sections: run (where results go, where to stop), execution (CPUs, lanes, scratch; never part of any scientific
identity), input (VCF or simulation, contigs), quality (evidence QC and read calibration), genetic_map (inference
prior), pedigree, family_phase, recombination, export (the final founder and sample VCFs), simulation (+ reads;
generating parameters only). Stage-internal constants are not settings: they are the defaults of each stage's own
config (stages/configure.py maps these sections into them).

The pipeline reads only the observations (VCF/BCF, or the simulated reads) and the genetic map. Sample names
are identifiers. The settings loader rejects sample-metadata sections.

Every field carries its kind (how TOML values are parsed and checked) and a one-line description (docs are
generated from it). Paths are resolved against the working directory when the configuration is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


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
            provenance="P0 ENG-09/ENG-10; WP_I1 section 4.1",
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
        "Contigs reconstructed concurrently; auto = one lane per 8 "
        "cores, paced by measured memory pressure; 1 = one contig at a time with all cores.",
        affects_results=False,
    )
    memory_limit: int | str = setting(
        "auto",
        "memory",
        "Capacity for measured job memory pressure: a size such as "
        '"200G" (binary units), or auto = job cgroup capacity. SOFT/HARD '
        "thresholds and sampling interval are in execution/config.py; affects scheduling only.",
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
        "Observations: an indexed VCF/BCF with FORMAT/AD, or a simulation from founder templates.",
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
    rate_is_measured: bool = setting(
        False,
        "bool",
        "rate_cm_per_mb is this organism's genome-wide average: the "
        "pedigree's map-length rule counts chromosome lengths from it where no map "
        "covers them. Off: the rate is a placeholder and only a supplied map's span "
        "counts.",
    )


@dataclass(frozen=True)
class PedigreeSection:
    evidence_calibration: str = setting(
        "predictive",
        "str",
        "Cross-chromosome pedigree evidence scaling: predictive, or off for unscaled evidence.",
        ("predictive", "off"),
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
    error_rate: float = setting(0.02, "float", "Generating sequencing error rate.")
    depth_cv: float = setting(0.0, "float", "Between-sample depth coefficient of variation.")
    heterozygote_alt_probability: float = setting(0.5, "float", "Alternate-read probability in heterozygotes.")
    dropout_fraction: float = setting(0.0, "float", "Contiguous missing-marker fraction per sample.")
    read_length_bp: int = setting(
        0, "int", "Read length; > 0 samples paired-end fragments shared by neighbouring sites."
    )
    fragment_mean_bp: float = setting(375.0, "float", "Mean fragment length (read_length_bp > 0).")
    fragment_sd_bp: float = setting(150.0, "float", "Fragment length standard deviation.")


@dataclass(frozen=True)
class SimulationSection:
    seed: int | str = setting(400, "seed", 'Simulation seed, or "random" (the realised seed is recorded).')
    templates: Path | None = setting(None, "path", "Directory of chromosome NPZ founder templates.")
    cohorts: tuple[int, ...] = setting((20, 100, 200), "ints", "Individuals in successive cohorts.")
    depth: float = setting(5.0, "float", "Mean read depth.")
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
        "generation, observed); replaces cohorts and the random matings.",
    )
    reads: SimulationReadsSection = section(SimulationReadsSection, "Generating read model.")


@dataclass(frozen=True)
class RelatednessSection:
    """Select optional pair output without changing the family model."""

    pairwise_ibd: bool = setting(
        False, "bool", "Write pairwise IBD reports (O(N² L)); off by default.", affects_results=False
    )


@dataclass(frozen=True)
class Settings:
    """The resolved configuration of one run (see the module docstring)."""

    version: int = setting(4, "int", "Settings schema version (4).", affects_results=False)
    preset: str = setting("", "str", "Built-in base configuration: tromau or generic.", ("", "tromau", "generic"))
    run: RunSection = section(RunSection, "Where results go and where to stop.")
    execution: ExecutionSection = section(ExecutionSection, "CPUs, lanes and scratch (never part of results).")
    input: InputSection = section(InputSection, "Observations and contigs.")
    quality: QualitySection = section(QualitySection, "Evidence quality control and read model.")
    genetic_map: GeneticMapSection = section(GeneticMapSection, "Inference genetic map.")
    relatedness: RelatednessSection = section(RelatednessSection, "Optional relatedness output selection.")
    pedigree: PedigreeSection = section(PedigreeSection, "Pedigree inference.")
    family_phase: FamilyPhaseSection = section(FamilyPhaseSection, "Family phase.")
    recombination: RecombinationSection = section(RecombinationSection, "Recombination maps.")
    export: ExportSection = section(ExportSection, "Final products: founder and sample VCFs.")
    simulation: SimulationSection = section(SimulationSection, "Simulation (input kind = simulation).")

    @property
    def serial(self) -> bool:
        """Contigs one at a time ([execution] serial, implied by max_contigs > 0)."""
        return self.execution.serial or self.execution.max_contigs > 0

    def site_filter(self) -> bool:
        """The site filter after resolving auto (on for VCF input, off for simulations)."""
        value = self.quality.site_filter
        return self.input.kind == "vcf" if value == "auto" else value == "on"

    def sample_filter(self) -> bool:
        value = self.quality.sample_filter
        return self.input.kind == "vcf" if value == "auto" else value == "on"


def schema_fields(cls) -> list:
    """The configuration fields of a section class, in declaration order."""
    return [item for item in fields(cls) if "kind" in item.metadata]


def to_dict(value) -> Any:
    """Plain data (paths as strings, None as "") of a config object."""
    if hasattr(value, "__dataclass_fields__"):
        return {item.name: to_dict(getattr(value, item.name)) for item in schema_fields(type(value))}
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
    "RelatednessSection",
    "PedigreeSection",
    "FamilyPhaseSection",
    "RecombinationSection",
    "ExportSection",
    "SimulationReadsSection",
    "SimulationSection",
    "schema_fields",
    "to_dict",
    "reference_rows",
]
