"""Command-line configuration for the canonical, checkpointed workflows."""
from pathlib import Path
import argparse
import json
import os
import runpy
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(value):
    return None if value in (None, '') else str(Path(value).expanduser().resolve())


def _setting(args, config, section, key, default=None):
    value = getattr(args, key, None)
    return config.get(section, {}).get(key, default) if value is None else value


def _shared_family_setting(value):
    """Match the map pipeline's boolean spellings without importing its kernels."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        setting = value.strip().lower()
        if setting in ('1', 'true', 'yes', 'on'):
            return True
        if setting in ('0', 'false', 'no', 'off'):
            return False
    raise ValueError('shared_family_evidence must be boolean (1/0, true/false, yes/no or on/off)')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Missing-aware haplotype reconstruction for experimental crosses.'
    )
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in (
        ('simulate', 'Simulate a known pedigree and reconstruct its haplotypes.'),
        ('reconstruct', 'Reconstruct an indexed biallelic AD VCF/BCF without cross-specific metadata.'),
        ('astcal', 'Reconstruct the AstCal × AulStu cross.'),
        ('tropheops', 'Reconstruct the Tropheops cross.'),
        ('recombination', 'Estimate maps from completed final-phase checkpoints.')):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            '--config',
            type=Path,
            help='TOML configuration; paths are relative to the current directory.'
        )
        command.add_argument('--output', help='Run directory containing outputs, logs and checkpoints.')
        command.add_argument('--checkpoints', help='Checkpoint directory; default OUTPUT/checkpoints.')
        command.add_argument(
            '--cores',
            type=int,
            help='Total process/thread ceiling; default current CPU affinity.'
        )
        command.add_argument('--recombination-map', help='PLINK/Beagle cumulative-cM input map.')
        command.add_argument('--rate-cm-per-mb', type=float, help='Fallback rate, default 5.0 cM/Mb.')
        command.add_argument('--shared-family-evidence', action=argparse.BooleanOptionalAction, default=None,
                             help='Use shared-family phase-error evidence in map generation; default on.')
        if name != 'recombination':
            command.add_argument('--assembly-model', choices=('dense', 'structured'),
                                 help='Dense cubic transitions (default) or near-quadratic structured transitions; independent of --assembly-search.')
            command.add_argument('--assembly-search', choices=('bounded', 'broad'),
                                 help='Bounded panel search (default, 16 full scores per category) or the optimized broader search with more full refits.')
            command.add_argument('--founder-refinement', choices=('on', 'off'),
                                 help='Refine original local-row choices after each final L1-L4 assembly level; default on. Excludes the two L1/L2 feedback passes.')
            command.add_argument('--discovery-search', choices=('standard', 'batched'),
                                 help='block discovery search, independent of assembly; default standard. Batched is experimental.')
            command.add_argument('--feedback-selection', choices=('balanced', 'strict'),
                                 help='Selection after each L1/L2 feedback round: balanced (default) or protected-backbone strict rescue.')
            command.add_argument('--contigs', nargs='+', help='Physical contigs to analyze, in input order.')
            command.add_argument(
                '--vcf',
                help='Input VCF/BCF; optional when simulation templates are supplied.'
            )
        if name == 'simulate':
            command.add_argument('--process-contigs', nargs='+',
                                 help='Process a chromosome shard from globally cached simulation inputs; does not change --contigs.')
            command.add_argument('--stop-after-stage', choices=('block_discovery', 'painting'),
                                 help='Stop cleanly at a checkpoint boundary, before genome-wide inference.')
            command.add_argument('--seed', type=int, help='Simulation seed, default 400.')
            command.add_argument('--depth', type=float, help='Mean simulated read depth, default 5.')
            command.add_argument('--templates', help='Directory of chromosome NPZ founder templates.')
            command.add_argument('--backcross-fraction', type=float, help='Fraction of later matings with an actual parent, default 0.')
            command.add_argument('--backcross-start-generation', type=int, help='First backcross cohort number, default 3.')
            command.add_argument('--observe-generations', type=int, nargs='+', help='Only sequence these cohort numbers, e.g. 4 5 6.')
            command.add_argument('--observed-fraction', type=float, help='Random fraction of eligible individuals sequenced, default 1.')
            command.add_argument('--generating-error-rate', type=float, help='Sequencing generator error, default 0.02; inference is unchanged.')
            command.add_argument('--depth-cv', type=float, help='Between-sample depth coefficient of variation, default 0.')
            command.add_argument('--heterozygote-alt-probability', type=float, help='Generating alternate-read probability in heterozygotes, default 0.5.')
            command.add_argument('--dropout-fraction', type=float, help='Contiguous missing-marker fraction per sample, default 0.')
            command.add_argument(
                '--generations',
                type=int,
                nargs='+',
                help='Individuals in successive cohorts, default 20 100 200.'
            )
            command.add_argument(
                '--generating-map',
                help='Optional simulation-generating map, independent of inference.'
            )
            command.add_argument(
                '--generating-rate-cm-per-mb',
                type=float,
                help='Generating fallback rate, default 5.0.'
            )
        elif name == 'reconstruct':
            command.add_argument('--eligibility', help='Optional JSON sample eligibility/chronology constraints.')
            command.add_argument('--stop-after-stage', choices=('block_discovery', 'painting'),
                                 help='Stop at a checkpoint boundary before genome-wide inference.')
        elif name != 'recombination':
            command.add_argument('--metadata', help='Cross-design metadata workbook.')
            command.add_argument('--metadata-sheet', help='Workbook sheet, default main_data.')
    evaluate = commands.add_parser(
        'evaluate',
        help='Evaluate a completed simulated run against cached truth.'
    )
    evaluate.add_argument('--output', required=True, help='Completed simulation run directory.')
    evaluate.add_argument('--cores', type=int, default=None)
    evaluate.add_argument('--checkpoints', help='Checkpoint root; default OUTPUT/checkpoints.')
    evaluate.add_argument('--contigs', nargs='+')
    evaluate.add_argument('--stages', nargs='+', default=['available'],
                          help='available (latest products), all, or descriptive founder/downstream stages.')
    evaluate.add_argument('--feedback-selection', choices=('balanced', 'strict'), default='balanced')
    export = commands.add_parser('export', help='Export released founder tracks and sample phase.')
    export.add_argument('--output', required=True, help='Existing run directory.')
    export.add_argument('--destination', required=True, help='New/empty export directory.')
    export.add_argument('--checkpoints')
    export.add_argument('--contigs', nargs='+')
    export.add_argument('--products', nargs='+', choices=('founders', 'samples'), default=['founders', 'samples'])
    export.add_argument('--format', choices=('bcf', 'vcf'), default='bcf')
    export.add_argument('--vcf', help='Original general-reconstruct input, normally recovered from its identity.')
    export.add_argument('--synthetic-alleles', action='store_true',
                        help='Simulation only: A/C labels for abstract 0/1, not biological reference bases.')
    export.add_argument('--cores', type=int, default=None)
    example = commands.add_parser('example', help='Create self-contained synthetic templates and a runnable config.')
    example.add_argument('--output', required=True, help='New/empty example directory.')
    example.add_argument('--seed', type=int, default=400)
    example.add_argument('--contigs', type=int, default=3)
    example.add_argument('--sites', type=int, default=1200)
    example.add_argument('--length-bp', type=int, default=12_000_000)
    status = commands.add_parser('status', help='Read lightweight run records without loading checkpoints.')
    status.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'example':
        from .simulation.example import create_example
        print(create_example(args.output, seed=args.seed, contigs=args.contigs,
                             sites=args.sites, length_bp=args.length_bp))
        return 0
    if args.command == 'status':
        from .core.run_record import run_status
        print(json.dumps(run_status(args.output), indent=2))
        return 0
    if args.command == 'export':
        if args.cores is not None:
            os.environ['NUMBA_NUM_THREADS'] = str(args.cores)
        from .workflows.export import export_run
        export_run(args.output, args.destination, checkpoints=args.checkpoints, contigs=args.contigs,
                   products=args.products, file_format=args.format, vcf=args.vcf,
                   synthetic_alleles=args.synthetic_alleles, cores=args.cores)
        return 0
    if args.command == 'evaluate':
        os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / 'work' / 'cache' / 'matplotlib'))
        if args.cores is not None:
            os.environ['NUMBA_NUM_THREADS'] = str(args.cores)
        from.simulation.evaluation import evaluate_run
        report = evaluate_run(args.output, checkpoints=args.checkpoints, cores=args.cores,
                              stages=args.stages, contigs=args.contigs,
                              feedback_selection=args.feedback_selection)
        print(json.dumps(report['pedigree'], indent=2))
        return 0
    config = {}
    if args.config is not None:
        with args.config.open('rb') as handle:
            config = tomllib.load(handle)
    if 'impute_missing' in config.get('refinement', {}):
        parser.error('[refinement].impute_missing has been retired; remove this setting. '
                     'family phase preserves called genotypes and missingness and does not publish '
                     'a separate imputed-allele product.')
    cores = _setting(args, config, 'run', 'cores', len(os.sched_getaffinity(0)))
    if cores == 'auto':
        cores = len(os.sched_getaffinity(0))
    cores = int(cores)
    if not 1 <= cores <= len(os.sched_getaffinity(0)):
        parser.error('--cores must fit the current CPU affinity')
    try:
        shared = _shared_family_setting(_setting(args, config, 'recombination', 'shared_family_evidence',
                                               os.environ.get('BHD_RECOMBINATION_SHARED_FAMILY', '1')))
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == 'simulate':
        shard_env = os.environ.get('BHD_SIM_CONTIGS')
        process_contigs = _setting(args, config, 'run', 'process_contigs',
                                None if shard_env is None else shard_env.split(','))
        if process_contigs is not None:
            if (not isinstance(process_contigs, list) or not process_contigs
                    or any(not isinstance(name, str) or not name or name != name.strip() or ',' in name
                           for name in process_contigs)
                    or len(set(process_contigs)) != len(process_contigs)):
                parser.error('process_contigs must be a nonempty list of unique exact contig names')
        stop_after_stage = _setting(
            args,
            config,
            'run',
            'stop_after_stage',
            os.environ.get('BHD_SIM_STOP_AFTER_STAGE')
        )
        if stop_after_stage is not None and stop_after_stage not in ('block_discovery', 'painting'):
            parser.error('stop_after_stage must be block_discovery or painting')
    if args.command != 'recombination':
        if 'founder_scaling' in config.get('run', {}):
            parser.error(
                'founder_scaling was replaced by independent assembly_model and discovery_search settings'
            )
        assembly_model = _setting(
            args,
            config,
            'run',
            'assembly_model',
            os.environ.get('HAPLOTYPES_ASSEMBLY_MODEL', 'dense')
        )
        if assembly_model not in ('dense', 'structured'):
            parser.error('assembly_model must be dense or structured')
        assembly_search = _setting(
            args,
            config,
            'run',
            'assembly_search',
            os.environ.get('HAPLOTYPES_ASSEMBLY_SEARCH', 'bounded')
        )
        if assembly_search not in ('bounded', 'broad'):
            parser.error('assembly_search must be bounded or broad')
        founder_refinement = _setting(
            args,
            config,
            'run',
            'founder_refinement',
            os.environ.get('HAPLOTYPES_FOUNDER_REFINEMENT', 'on')
        )
        if founder_refinement not in ('on', 'off'):
            parser.error('founder_refinement must be on or off')
        discovery_search = _setting(
            args,
            config,
            'run',
            'discovery_search',
            os.environ.get('HAPLOTYPES_DISCOVERY_SEARCH', 'standard')
        )
        if discovery_search not in ('standard', 'batched'):
            parser.error('discovery_search must be standard or batched')
        feedback_selection = _setting(
            args,
            config,
            'run',
            'feedback_selection',
            os.environ.get('HAPLOTYPES_FEEDBACK_SELECTION', 'balanced')
        )
        if feedback_selection not in ('balanced', 'strict'):
            parser.error('feedback_selection must be balanced or strict')
        os.environ['HAPLOTYPES_ASSEMBLY_MODEL'] = assembly_model
        os.environ['HAPLOTYPES_ASSEMBLY_SEARCH'] = assembly_search
        os.environ['HAPLOTYPES_FOUNDER_REFINEMENT'] = founder_refinement
        os.environ['HAPLOTYPES_DISCOVERY_SEARCH'] = discovery_search
        os.environ['HAPLOTYPES_FEEDBACK_SELECTION'] = feedback_selection
    seed = _setting(args, config, 'simulation', 'seed', 400)
    default_output=f'work/runs/seed_{seed}' if args.command=='simulate' else f'work/runs/{args.command}'
    output = Path(_setting(args, config, 'run', 'output', default_output)).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / 'work' / 'cache' / 'matplotlib'))
    for name in ('HAPLOTYPES_CONTIGS', 'HAPLOTYPES_VCF', 'HAPLOTYPES_TEMPLATES',
                 'HAPLOTYPES_METADATA', 'BHD_SIM_CONTIGS', 'BHD_SIM_STOP_AFTER_STAGE',
                 'HAPLOTYPES_ELIGIBILITY', 'HAPLOTYPES_STOP_AFTER_STAGE'):
        os.environ.pop(name, None)
    checkpoints = Path(_setting(args, config, 'run', 'checkpoints', output / 'checkpoints')).expanduser().resolve()
    rate = float(_setting(args, config, 'recombination', 'rate_cm_per_mb', 5.0))
    mapping = _path(_setting(args, config, 'recombination', 'recombination_map'))
    os.environ.update(BHD_NUM_PROCESSES=str(cores), NUMBA_NUM_THREADS=str(cores),
        BHD_RECOMBINATION_RATE_CM_PER_MB=str(rate), BHD_RECOMBINATION_SHARED_FAMILY='1' if shared else '0',
        HAPLOTYPES_OUTPUT_DIR=str(output),
        HAPLOTYPES_CHECKPOINT_DIR=str(checkpoints), HAPLOTYPES_LOG_DIR=str(output / 'logs'))
    if mapping:
        os.environ['BHD_RECOMBINATION_MAP'] = mapping
    else:
        os.environ.pop('BHD_RECOMBINATION_MAP', None)
    if args.command != 'recombination':
        contigs = _setting(args, config, 'inputs', 'contigs')
        if contigs:
            os.environ['HAPLOTYPES_CONTIGS'] = json.dumps(contigs)
        vcf = _path(_setting(args, config, 'inputs', 'vcf'))
        if vcf:
            os.environ['HAPLOTYPES_VCF'] = vcf
    if args.command == 'simulate':
        if process_contigs is not None:
            os.environ['BHD_SIM_CONTIGS'] = ','.join(process_contigs)
        if stop_after_stage is not None:
            os.environ['BHD_SIM_STOP_AFTER_STAGE'] = stop_after_stage
        os.environ.update(BHD_SIMULATION_SEED=str(seed), BHD_SIM_READ_DEPTH=str(_setting(args, config, 'simulation', 'depth', 5.0)),
            BHD_SIM_CHECKPOINT_DIR=str(checkpoints), BHD_SIM_OUTPUT_DIR=str(output),
            HAPLOTYPES_GENERATIONS=json.dumps(_setting(args, config, 'simulation', 'generations', [20, 100, 200])),
            BHD_SIMULATION_RECOMBINATION_RATE_CM_PER_MB=str(_setting(args, config, 'simulation', 'generating_rate_cm_per_mb', 5.0)))
        from .simulation.designs import SimulationDesign, ReadModel
        design = SimulationDesign(
            backcross_fraction=_setting(args, config, 'simulation', 'backcross_fraction', 0.0),
            backcross_start_generation=_setting(args, config, 'simulation', 'backcross_start_generation', 3),
            observe_generations=tuple(_setting(args, config, 'simulation', 'observe_generations', [])),
            observed_fraction=_setting(args, config, 'simulation', 'observed_fraction', 1.0))
        read_model = ReadModel(
            error_rate=_setting(args, config, 'simulation', 'generating_error_rate', 0.02),
            depth_cv=_setting(args, config, 'simulation', 'depth_cv', 0.0),
            heterozygote_alt_probability=_setting(args, config, 'simulation', 'heterozygote_alt_probability', 0.5),
            dropout_fraction=_setting(args, config, 'simulation', 'dropout_fraction', 0.0))
        os.environ['HAPLOTYPES_SIMULATION_DESIGN'] = json.dumps(design.record())
        os.environ['HAPLOTYPES_READ_MODEL'] = json.dumps(read_model.record())
        templates = _path(_setting(args, config, 'inputs', 'templates'))
        if templates:
            os.environ['HAPLOTYPES_TEMPLATES'] = templates
        generating_map = _path(_setting(args, config, 'simulation', 'generating_map'))
        if generating_map:
            os.environ['BHD_SIMULATION_RECOMBINATION_MAP'] = generating_map
        else:
            os.environ.pop('BHD_SIMULATION_RECOMBINATION_MAP', None)
        runpy.run_module('haplotype_reconstruction.workflows.simulation', run_name='__main__')
    elif args.command == 'reconstruct':
        if not vcf:
            parser.error('reconstruct requires --vcf or [inputs].vcf')
        eligibility = _path(_setting(args, config, 'inputs', 'eligibility'))
        if eligibility:
            os.environ['HAPLOTYPES_ELIGIBILITY'] = eligibility
        stop = _setting(args, config, 'run', 'stop_after_stage')
        if stop is not None:
            if stop not in ('block_discovery', 'painting'):
                parser.error('stop_after_stage must be block_discovery or painting')
            os.environ['HAPLOTYPES_STOP_AFTER_STAGE'] = stop
        runpy.run_module('haplotype_reconstruction.workflows.variants', run_name='__main__')
    elif args.command in ('astcal', 'tropheops'):
        metadata = _path(_setting(args, config, 'inputs', 'metadata'))
        if metadata:
            os.environ['HAPLOTYPES_METADATA'] = metadata
        os.environ['HAPLOTYPES_METADATA_SHEET'] = str(_setting(args, config, 'inputs', 'metadata_sheet', 'main_data'))
        runpy.run_module('haplotype_reconstruction.workflows.' + args.command, run_name='__main__')
    else:
        from.core.genetic_map import load_genetic_maps
        from.recombination.model import RecombinationMapConfig
        from.recombination.pipeline import run_from_checkpoints
        maps = load_genetic_maps(mapping, rate)
        run_from_checkpoints(checkpoints, output, n_workers=cores,
            config=RecombinationMapConfig(recombination_rate=rate / 1e8),
            genetic_maps=maps if maps.maps else None, shared_family_evidence=shared)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
