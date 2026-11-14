"""Command-line interface: `haplotypes <command>` (or `python -m haplotype_reconstruction <command>`).

    run [CONFIG]        the pipeline (VCF cross, generic VCF or simulation; options override CONFIG)
    status RUN          run records and checkpoint progress
    recombination RUN   the recombination stage of a run whose family phase is recorded (runs or resumes it)
    evaluate RUN        simulations: accuracy against the known truth
    export RUN DEST     re-export a run's founder and sample VCFs (and founder-track TSVs)
    example DEST        a small coalescent simulation configuration
    observations CONFIG export simulated observations and a bound truth sidecar
    init / evidence / relatedness / reconstruct / pedigree / phase / report: cuts of the same graph

Settings come from the configuration (see docs/configuration.md); scientific switches are changed with
`--set section.key=value`. The parser imports no numerical code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _add_run_options(command):
    command.add_argument(
        "config", nargs="?", type=Path, help="TOML configuration (paths relative to the working directory)."
    )
    command.add_argument("--preset", choices=("tromau", "generic"), help="Built-in base configuration.")
    command.add_argument("--vcf", help="Input VCF/BCF with FORMAT/AD ([input] vcf).")
    command.add_argument(
        "--templates",
        help='Legacy alias for [simulation.source] kind = "file", file = DIR; sets simulation input.',
    )
    command.add_argument("--output", help="Run directory ([run] output).")
    command.add_argument("--checkpoints", help="Checkpoint directory ([run] checkpoints).")
    command.add_argument("--cores", type=int, help="CPU ceiling ([execution] cores; default: affinity).")
    command.add_argument(
        "--memory-limit",
        help="Capacity for measured job memory pressure, e.g. 200G (binary units) "
        "([execution] memory_limit; auto = job cgroup capacity).",
    )
    command.add_argument(
        "--contigs",
        nargs="+",
        help="Contigs in order ([input] contigs); run selects a subset of a configured list, "
        "or supplies the initial list when none is configured.",
    )
    command.add_argument("--stop-after", choices=("relatedness", "discovery", "painting"), help="[run] stop_after.")
    command.add_argument("--serial", action="store_true", default=None, help="[execution] serial.")
    command.add_argument("--max-contigs", type=int, help="[execution] max_contigs (implies --serial).")
    command.add_argument("--contig-lanes", type=int, help="Concurrent contig lanes ([execution] contig_lanes).")
    command.add_argument("--scratch", help="Mapped scratch and kernel cache base ([execution] scratch).")
    command.add_argument(
        "--infer-rate",
        action="store_true",
        default=None,
        help="Use the recombination rate estimated from relatives for contigs without a map file when it "
        "differs from the supplied rate ([genetic_map] infer_rate).",
    )
    command.add_argument("--tie-reverse", action="store_true", help="Reverse arbitrary tie conventions (developer).")
    command.add_argument(
        "--tie-keep", nargs="+", default=(), metavar="NAMES", help="Keep these tie conventions (developer)."
    )
    command.add_argument("--tie-census", type=Path, metavar="DIR", help="Record tie diagnostics (developer).")
    command.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override one setting (repeatable), e.g. quality.read_calibration=false.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="haplotypes",
        description="Founder-haplotype reconstruction, pedigree, "
        "phase and recombination maps for experimental crosses.",
    )
    from .. import __version__

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "Initialize if needed, then run or resume the graph."),
        ("init", "Validate inputs and initialize the run manifest."),
        ("evidence", "Record one contig's evidence."),
        ("relatedness", "Gather relatedness from every contig's evidence."),
        ("reconstruct", "Reconstruct and refit one contig; concurrent invocations share unclaimed units."),
        ("pedigree", "Gather the genome-wide pedigree."),
        ("phase", "Phase, map and export one contig."),
        ("report", "Publish tables from available records."),
    ):
        command = commands.add_parser(name, help=help_text)
        _add_run_options(command)
        if name in ("evidence", "reconstruct", "phase"):
            command.add_argument("--contig", required=True, help="Configured contig for this scatter step.")

    status = commands.add_parser("status", help="Run records and checkpoint progress (no payload decoding).")
    status.add_argument("run_directory", type=Path, metavar="RUN")
    status.add_argument("--checkpoints", type=Path)
    status.add_argument("--contigs", nargs="+")
    status.add_argument("--json", action="store_true", help="Print the report as JSON.")

    recombination = commands.add_parser(
        "recombination",
        help="Run or resume the recombination stage of a run whose family phase is recorded "
        "and republish its tables (a recorded stage keeps its settings: changed settings need a new output root).",
    )
    recombination.add_argument("run_directory", type=Path, metavar="RUN")
    recombination.add_argument("--order-breakpoints", help="[recombination] order_breakpoints TSV.")
    recombination.add_argument("--cores", type=int)
    recombination.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="recombination.KEY=VALUE"
    )

    evaluate = commands.add_parser("evaluate", help="Evaluate a simulation run against its truth.")
    evaluate.add_argument("run_directory", type=Path, metavar="RUN")
    evaluate.add_argument(
        "--stages", nargs="+", default=["available"], help="available (latest products), all, or stage names."
    )
    evaluate.add_argument("--contigs", nargs="+")
    evaluate.add_argument("--cores", type=int)

    evaluate.add_argument("--truth", type=Path, help="Bound observation-package truth directory.")
    evaluate.add_argument("--output", type=Path, help="Evaluation destination with --truth.")
    evaluate.add_argument("--release-tree", type=Path, help="Source root of the run's release, with --truth.")
    evaluate.add_argument("--release-python", type=Path, help="Interpreter for the release reader, with --truth.")
    evaluate.add_argument("--timeout", type=float, default=600.0, help="Release-reader timeout in seconds.")

    observations = commands.add_parser("observations", help="Publish simulated BCF observations and bound truth.")
    observations.add_argument("config", type=Path, metavar="CONFIG")
    observations.add_argument("--output", type=Path, required=True, metavar="DIR")
    observations.add_argument("--preset", choices=("tromau", "generic"))
    observations.add_argument("--cores", type=int)
    observations.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")

    export = commands.add_parser(
        "export",
        help="Write a finished run's founder VCF and phased sample VCF (the files of the run's export "
        "stage, OUTPUT/export) to a new directory, optionally with per-contig founder-track CSVs.",
    )
    export.add_argument("run_directory", type=Path, metavar="RUN")
    export.add_argument("destination", type=Path, metavar="DEST", help="New or empty directory.")
    export.add_argument(
        "--products",
        nargs="+",
        choices=("founders", "samples", "tracks"),
        default=["founders", "samples"],
        help="founders: per-contig founder VCF (haploid GT, one column c<component>_H<row> per "
        "founder row); samples: phased sample VCF (GT:PS); tracks: per-contig founder-track "
        "CSV.gz (contig, component, pos, h0, h1, ...).",
    )
    export.add_argument(
        "--format",
        choices=("vcf", "bcf"),
        default="vcf",
        help="vcf: bgzipped VCF with a tabix index (CSI beyond 2^29 bp); bcf: BCF with a CSI index.",
    )
    export.add_argument(
        "--full-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write every input biallelic SNP that reached the site filter (FILTER PASS for kept "
        "sites, the rule codes r1-r5 for removed ones); --no-full-length writes the kept sites only.",
    )
    export.add_argument("--contigs", nargs="+")
    export.add_argument("--cores", type=int)

    example = commands.add_parser("example", help="Create a coalescent count-mode simulation configuration.")
    example.add_argument("destination", type=Path, metavar="DEST")
    example.add_argument("--seed", type=int, default=400)
    example.add_argument("--contigs", type=int, default=3, help="Number of synthetic contigs.")
    example.add_argument("--sites", type=int, default=1200)
    example.add_argument("--length-bp", type=int, default=12_000_000)

    return parser


def run_options(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Command-line flags of `run` as configuration sections (TOML-typed values)."""
    options: dict[str, dict[str, Any]] = {}

    def put(section, key, value):
        if value is not None:
            options.setdefault(section, {})[key] = value

    if args.templates is not None:
        put("input", "kind", "simulation")
        put("simulation", "templates", args.templates)
    elif args.vcf is not None:
        put("input", "kind", "vcf")
    put("input", "vcf", args.vcf)
    put("input", "contigs", args.contigs)
    put("run", "output", args.output)
    put("run", "checkpoints", args.checkpoints)
    put("run", "stop_after", args.stop_after)
    put("execution", "cores", args.cores)
    put("execution", "memory_limit", args.memory_limit)
    put("execution", "serial", args.serial)
    put("execution", "max_contigs", args.max_contigs)
    put("execution", "contig_lanes", args.contig_lanes)
    put("execution", "scratch", args.scratch)
    put("genetic_map", "infer_rate", args.infer_rate)
    return options


def _command_workflow(args: argparse.Namespace) -> int:
    from .run import execute, run_paths
    from .settings.load import load_settings

    options = run_options(args)
    selected = None
    if args.command == "run" and args.contigs is not None:
        selected = tuple(args.contigs)
        options["input"].pop("contigs")
        if not options["input"]:
            del options["input"]
    settings = load_settings(
        args.config, preset=args.preset, options=options, overrides=args.overrides, fallback_contigs=selected
    )
    from dataclasses import replace

    from ..results.paths import active_exclusions
    from .settings.load import parse_override

    output = run_paths(settings).output
    recorded = args.config is not None and args.config.expanduser().resolve() == output / "config.toml"
    explicit = any(tuple(parse_override(value)[0]) == ("run", "exclude_contigs") for value in args.overrides)
    if recorded and not explicit:
        active = active_exclusions(output)
        if active is not None:
            settings = replace(settings, run=replace(settings.run, exclude_contigs=active))
    args.output = output
    execute(
        settings,
        command=args.command,
        contig=getattr(args, "contig", None),
        contigs=selected,
        tie_reverse=args.tie_reverse,
        tie_keep=tuple(args.tie_keep),
        tie_census=args.tie_census,
    )
    return 0


def _command_status(args: argparse.Namespace) -> int:
    from .status import format_status, run_status

    report = run_status(args.run_directory, checkpoints=args.checkpoints, contigs=args.contigs)
    print(json.dumps(report, indent=2) if args.json else format_status(report))
    return report.get("exit_code", 1) if report["errors"] else 0


def _command_recombination(args: argparse.Namespace) -> int:
    from ..core.errors import SettingsError
    from .run import _load_recorded_settings, rerun_recombination
    from .settings.load import parse_override

    for override in args.overrides:
        path, _ = parse_override(override)
        if len(path) != 2 or path[0] != "recombination":
            raise SettingsError("recombination accepts only --set recombination.KEY=VALUE")
    options: dict[str, dict[str, Any]] = {}
    if args.order_breakpoints is not None:
        options["recombination"] = {"order_breakpoints": str(Path(args.order_breakpoints).resolve())}
    settings = _load_recorded_settings(args.run_directory, args.cores, options=options, overrides=args.overrides)
    rerun_recombination(settings, args.run_directory)
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    if args.command in {"run", "init", "evidence", "relatedness", "reconstruct", "pedigree", "phase", "report"}:
        return _command_workflow(args)
    if args.command == "status":
        return _command_status(args)
    if args.command == "recombination":
        return _command_recombination(args)
    if args.command == "example":
        from .example import example_command

        return example_command(args)
    if args.command == "observations":
        from .run import observations_command

        return observations_command(args)
    if args.command == "evaluate":
        from .run import evaluate_command

        return evaluate_command(args)
    from .run import export_command

    return export_command(args)


def main(argv: list[str] | None = None) -> int:
    """Parse one command and map typed failures to the documented, stable exit codes."""
    import platform
    import sys

    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        print("haplotypes: Linux x86_64 is required by the tested execution platform", file=sys.stderr)
        return 4
    args = build_parser().parse_args(argv)

    from ..core.errors import InputError, ResourceError, SettingsError
    from ..engine.runtime import RunInterrupted, signal_boundary

    try:
        with signal_boundary(lambda: None):
            return _dispatch(args)
    except (KeyboardInterrupt, RunInterrupted):
        print("haplotypes: interrupted; recorded state is resumable", file=sys.stderr)
        return 130
    except SettingsError as error:
        print(f"haplotypes {args.command}: {error}", file=sys.stderr)
        return 2
    except InputError as error:
        print(f"haplotypes {args.command}: {error}", file=sys.stderr)
        return 3
    except ResourceError as error:
        print(f"haplotypes {args.command}: {error}", file=sys.stderr)
        return 4
    except Exception as error:
        # Importing the container before dispatch would load NumPy before the
        # runtime establishes its numerical policy in a fresh CLI interpreter.
        from ..store.container import CheckpointMismatch

        if isinstance(error, CheckpointMismatch):
            print(f"haplotypes {args.command}: {error}", file=sys.stderr)
            return 2
        from .run import error_report

        report = error_report(error, args)
        print(f"haplotypes {args.command}: {error}; report: {report}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "run_options", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
