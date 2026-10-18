"""Command-line interface: `haplotypes <command>` (or `python -m haplotype_reconstruction <command>`).

    run [CONFIG]        the pipeline (VCF cross, generic VCF or simulation; options override CONFIG)
    status RUN          run records and checkpoint progress
    recombination RUN   the recombination stage of a run whose family phase is recorded (runs or resumes it)
    evaluate RUN        simulations: accuracy against the known truth
    export RUN DEST     re-export a run's founder and sample VCFs (and founder-track TSVs)
    example DEST        a small synthetic example (templates + configuration)
    templates           simulation templates from a VCF (naive builder)
    batch-worker QUEUE  serve a run's shared batch queue with this allocation

Settings come from the configuration (see docs/configuration.md); scientific switches are changed with
`--set section.key=value`. The parser imports no numerical code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _add_run_options(command):
    command.add_argument("config", nargs="?", type=Path, help="TOML configuration (paths relative to the "
                         "working directory).")
    command.add_argument("--preset", choices=("tromau", "aulstu"), help="Built-in base configuration.")
    command.add_argument("--vcf", help="Input VCF/BCF with FORMAT/AD ([input] vcf).")
    command.add_argument("--templates", help="Simulate from these founder templates ([simulation] templates; "
                         "sets [input] kind = \"simulation\").")
    command.add_argument("--output", help="Run directory ([run] output).")
    command.add_argument("--checkpoints", help="Checkpoint directory ([run] checkpoints).")
    command.add_argument("--cores", type=int, help="CPU ceiling ([execution] cores; default: affinity).")
    command.add_argument("--memory-limit", help="Memory the run may plan to use at once, e.g. 200G "
                         "([execution] memory_limit; default: available at start minus headroom).")
    command.add_argument("--contigs", nargs="+", help="Contigs in order ([input] contigs).")
    command.add_argument("--stop-after", choices=("discovery", "painting"), help="[run] stop_after.")
    command.add_argument("--serial", action="store_true", default=None, help="[execution] serial.")
    command.add_argument("--max-contigs", type=int, help="[execution] max_contigs (implies --serial).")
    command.add_argument("--process-contigs", nargs="+", help="Shard: [execution] process_contigs.")
    command.add_argument("--batch-queue", help="[execution] batch_queue.")
    command.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                         help="Override one setting (repeatable), e.g. quality.read_calibration=false.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="haplotypes", description="Founder-haplotype reconstruction, pedigree, "
                                     "phase and recombination maps for experimental crosses.")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_run_options(commands.add_parser("run", help="Run or resume the pipeline."))

    status = commands.add_parser("status", help="Run records and checkpoint progress (no payload decoding).")
    status.add_argument("run_directory", type=Path, metavar="RUN")
    status.add_argument("--checkpoints", type=Path)
    status.add_argument("--contigs", nargs="+")
    status.add_argument("--json", action="store_true", help="Print the report as JSON.")

    recombination = commands.add_parser(
        "recombination", help="Run or resume the recombination stage of a run whose family phase is recorded "
        "and republish its tables (a recorded stage keeps its settings: changed settings need a new output root).")
    recombination.add_argument("run_directory", type=Path, metavar="RUN")
    recombination.add_argument("--order-breakpoints", help="[recombination] order_breakpoints TSV.")
    recombination.add_argument("--cores", type=int)
    recombination.add_argument("--set", dest="overrides", action="append", default=[],
                               metavar="recombination.KEY=VALUE")

    evaluate = commands.add_parser("evaluate", help="Evaluate a simulation run against its truth.")
    evaluate.add_argument("run_directory", type=Path, metavar="RUN")
    evaluate.add_argument("--stages", nargs="+", default=["available"],
                          help="available (latest products), all, or stage names.")
    evaluate.add_argument("--contigs", nargs="+")
    evaluate.add_argument("--cores", type=int)

    export = commands.add_parser(
        "export", help="Write a finished run's founder VCF and phased sample VCF (the files of the run's export "
        "stage, OUTPUT/export) to a new directory, optionally with per-component founder-track TSVs.")
    export.add_argument("run_directory", type=Path, metavar="RUN")
    export.add_argument("destination", type=Path, metavar="DEST", help="New or empty directory.")
    export.add_argument("--products", nargs="+", choices=("founders", "samples", "tracks"),
                        default=["founders", "samples"],
                        help="founders: per-contig founder VCF (haploid GT, one column c<component>_H<row> per "
                        "founder row); samples: phased sample VCF (GT:PS); tracks: per-component founder-track "
                        "TSV.gz (CHROM POS H0 H1 ...).")
    export.add_argument("--format", choices=("vcf", "bcf"), default="vcf",
                        help="vcf: bgzipped VCF with a tabix index (CSI beyond 2^29 bp); bcf: BCF with a CSI index.")
    export.add_argument("--full-length", action=argparse.BooleanOptionalAction, default=True,
                        help="Write every input biallelic SNP that reached the site filter (FILTER PASS for kept "
                        "sites, the rule codes r1-r5 for removed ones); --no-full-length writes the kept sites only.")
    export.add_argument("--vcf", help="Original input VCF/BCF (normally recovered from the run; simulations need "
                        "none: their abstract alleles are written as A/C).")
    export.add_argument("--contigs", nargs="+")
    export.add_argument("--cores", type=int)

    example = commands.add_parser("example", help="Create synthetic templates and a runnable configuration.")
    example.add_argument("destination", type=Path, metavar="DEST")
    example.add_argument("--seed", type=int, default=400)
    example.add_argument("--contigs", type=int, default=3, help="Number of synthetic contigs.")
    example.add_argument("--sites", type=int, default=1200)
    example.add_argument("--length-bp", type=int, default=12_000_000)

    templates = commands.add_parser("templates", help="Build simulation templates from a VCF (naive builder).")
    templates.add_argument("--vcf", required=True)
    templates.add_argument("--output", required=True, type=Path)
    templates.add_argument("--contigs", nargs="+")
    templates.add_argument("--cores", type=int)

    helper = commands.add_parser("batch-worker", help="Serve a run's shared batch queue with this allocation.")
    helper.add_argument("queue", type=Path, metavar="QUEUE")
    helper.add_argument("--cores", type=int)
    helper.add_argument("--idle-seconds", type=int, default=120)
    return parser


def run_options(args) -> dict:
    """Command-line flags of `run` as configuration sections (TOML-typed values)."""
    options = {}

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
    put("execution", "process_contigs", args.process_contigs)
    put("execution", "batch_queue", args.batch_queue)
    return options


def _command_run(args, parser) -> int:
    from .config import ConfigError, load_config
    try:
        config = load_config(args.config, preset=args.preset, options=run_options(args), overrides=args.overrides)
    except ConfigError as error:
        parser.error(str(error))
    from .pipeline.driver import check_run, run_pipeline
    try:   # fail-fast checks (inputs, workbook, eligibility, checkpoint directory, shard preconditions)
        prepared = check_run(config)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"haplotypes run: {error}\n")
    run_pipeline(config, prepared)
    return 0


def _command_status(args) -> int:
    from .pipeline.status import format_status, run_status
    report = run_status(args.run_directory, checkpoints=args.checkpoints, contigs=args.contigs)
    print(json.dumps(report, indent=2) if args.json else format_status(report))
    return 1 if report["errors"] else 0


def _command_batch_worker(args, parser) -> int:
    from . import execution
    available = execution.affinity_cpus()
    cores = available if args.cores is None else args.cores
    if not 1 <= cores <= available or args.idle_seconds < 1:
        parser.error("batch-worker cores must fit the CPU affinity and idle-seconds must be positive")
    from .execution.batch_queue import serve
    from .execution.kernel_cache import persistent_kernel_cache
    from .pipeline.driver import PRELOAD_MODULES
    execution.start_run(cores, preload=PRELOAD_MODULES)
    with persistent_kernel_cache(cores):
        serve(args.queue.resolve(), cores, args.idle_seconds)
    return 0


def _command_recombination(args, parser) -> int:
    from .config import ConfigError, load_config
    overrides = list(args.overrides)
    if any(not text.startswith("recombination.") for text in overrides):
        parser.error("recombination accepts only --set recombination.KEY=VALUE")
    if args.order_breakpoints is not None:
        overrides.append(f'recombination.order_breakpoints="{Path(args.order_breakpoints).resolve()}"')
    if args.cores is not None:
        overrides.append(f"execution.cores={args.cores}")
    try:
        config = load_config(args.run_directory / "config.toml", overrides=overrides)
    except (ConfigError, FileNotFoundError) as error:
        parser.error(str(error))
    from .pipeline.driver import rerun_recombination
    from .store import CheckpointMismatch
    try:
        rerun_recombination(config, args.run_directory)
    except (CheckpointMismatch, FileNotFoundError) as error:
        parser.exit(2, f"haplotypes recombination: {error}\n")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _command_run(args, parser)
    if args.command == "status":
        return _command_status(args)
    if args.command == "batch-worker":
        return _command_batch_worker(args, parser)
    if args.command == "recombination":
        return _command_recombination(args, parser)
    # Commands of the reporting layer (evaluation, export, example, template building).
    if args.command == "evaluate":
        from .evaluation import evaluate_command as command_function
    elif args.command == "export":
        from .export import export_command as command_function
    elif args.command == "example":
        from .templates import example_command as command_function
    else:
        from .templates import templates_command as command_function
    return int(command_function(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
