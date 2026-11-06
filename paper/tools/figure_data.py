#!/usr/bin/env python3
"""Extract manuscript summaries from completed run directories into data/*.csv|json.

Reads only published tables/VCFs, evaluation reports, configs and timing receipts;
never imports or runs the pipeline. All input locations are command-line arguments.
Use --costs-only to refresh historical costs; --depth-run and --matrix44 preserve
legacy summaries while extracting new depth evidence and checking retained scores.
The plotting script reads only the compact output directory. See figures/CAPTIONS.md
for denominators, aggregation rules and the deliberately limited stage coverage.

Example (paths supplied by the caller):
  python -P paper/tools/figure_data.py --matrix42 MATRIX42 --matrix41 MATRIX41 \
      --probes PROBES --node-logs NODE_LOGS \
      --processor 'PREFIX_A=Sapphire Rapids:112' \
      --processor 'PREFIX_B=Ice Lake:76' --mosaic-chromosome chr1 --out paper/data

Replace the path and prefix placeholders with the caller's input locations and
receipt host prefixes. Processor mappings are required; none are built in or
written into the summaries. Names and full-node core counts above reproduce the
current tables when paired with their corresponding receipt prefixes.

VCF agreement uses streaming SHA-256 of the #CHROM header and all records, excluding
only ## metadata; it includes missingness, founder labels, GT phase and PS. Other
agreement checks use named released tables, relative tolerance 1e-9 plus absolute tolerance 1e-8 for numeric
cells, exact discrete/string cells, and omit elapsed_seconds. Align rows and sample
columns; canonicalize family IDs, unordered parents and painting-track exchange. No checkpoints read.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
DESIGNS = ['std', 'std_n80', 'stress_n80', 'backcross', 'bc8', 'longbc', 'f3',
           'f3x8', 'longf3', 'famhs', 'hsrand', 'sibrich', 'seed', 'tl', 'k30',
           'actm3', 'tromau']
VARIANTS = ['ref', 'perm_s1', 'iid12_s1', 'iid9_s1', 'val12_s1', 'vcfperm_s1', 'rev']


def natural(value):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', str(value))]


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle, delimiter='\t' if path.suffix == '.tsv' else ','))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(out, name, records):
    if not records:
        raise ValueError(f'No records for {name}')
    with (out / name).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    print(f'{name}: {len(records)} rows', flush=True)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else ''


def run_path(matrix, version, case):
    return matrix / 'runs' / f'e2e{version.replace(".", "")}f_{case}'


def source(version, case, name):
    return f'matrix{version.replace(".", "")}/runs/e2e{version.replace(".", "")}f_{case}/{name}'


def evaluation(run, version, spec):
    summary = read_json(run / 'evaluation/summary.json')
    founders = rows(run / 'evaluation/founders.csv')
    components = rows(run / 'evaluation/founder_components.csv')
    stages = sorted({r['stage'] for r in founders})
    chromosomes = summary['chromosomes']
    total = lambda key: sum(int(r[key]) for r in chromosomes)
    final = 'founder_refit' if 'founder_refit' in stages else stages[-1]
    f = [r for r in founders if r['stage'] == final]
    founder_total = lambda key: sum(int(r[key]) for r in f)
    alleles = founder_total('called_alleles')
    errors = founder_total('one_to_one_called_allele_errors')
    p = summary['pedigree']
    result = dict(case=spec['case'], design=spec['design'], variant=spec['variant'],
                  version=version, samples=summary['samples'], chromosomes=len(chromosomes),
                  founder_stage=final, founder_called_alleles=alleles,
                  founder_one_to_one_errors=errors,
                  founder_errors_per_million=ratio(1e6 * errors, alleles),
                  founder_missing_alleles=founder_total('missing_alleles'),
                  called_genotypes=total('called_genotypes'), genotype_errors=total('genotype_errors'),
                  genotype_error_rate=ratio(total('genotype_errors'), total('called_genotypes')),
                  sample_called_alleles=total('called_alleles'), sample_total_alleles=total('total_alleles'),
                  called_share=ratio(total('called_alleles'), total('total_alleles')),
                  switch_errors=total('switch_errors'),
                  phased_true_heterozygotes=total('phased_true_heterozygotes'),
                  correct_true_heterozygotes=total('correct_true_heterozygotes'),
                  switches_per_phased_heterozygote=ratio(total('switch_errors'), total('phased_true_heterozygotes')),
                  phased_share=ratio(total('phased_true_heterozygotes'), total('correct_true_heterozygotes')),
                  released_phase_sets=total('released_phase_sets'),
                  true_edges=p['true_edges'], correct_edges=p['correct_edges'],
                  missed_edges=p['missing_edges'], extra_edges=p['extra_edges'],
                  correct_parent_states=p['correct_parent_states'],
                  source=source(version, spec['case'], 'evaluation/{summary.json,founders.csv}'))
    fields = ['contig', 'stage', 'component', 'first_position', 'last_position', 'founder_rows',
              'truth_founders', 'markers', 'called_alleles', 'missing_alleles',
              'called_allele_errors', 'one_to_one_called_allele_errors',
              'unmatched_founder_rows', 'unmatched_truth_founders', 'unmatched_called_alleles',
              'segment_allele_errors', 'chimeric_segment_charges', 'split_charged_segment_cost']
    compact = []
    for r in components:
        c = dict(case=spec['case'], design=spec['design'], variant=spec['variant'], version=version)
        c.update({k: r[k] for k in fields})
        c['one_to_one_errors_per_million'] = ratio(1e6 * int(r['one_to_one_called_allele_errors']), int(r['called_alleles']))
        c['source'] = source(version, spec['case'], 'evaluation/founder_components.csv')
        compact.append(c)
    return result, compact, stages


def time_receipt(path):
    text = path.read_text(encoding='utf-8')
    elapsed = re.findall(r'Elapsed \(wall clock\) time .*?: ([\d:.]+)', text)
    rss = re.findall(r'Maximum resident set size \(kbytes\): (\d+)', text)
    status = re.findall(r'Exit status: (\d+)', text)
    if not elapsed or not status or status[-1] != '0':
        return None
    seconds = 0.0
    for part in elapsed[-1].split(':'):
        seconds = seconds * 60 + float(part)
    return dict(wall_seconds=round(seconds, 6), peak_rss_kib=int(rss[-1]), exit_status=0)


def processor_argument(value):
    """Parse an explicit host-prefix to processor-name/full-node-core mapping."""
    try:
        prefix, description = value.split('=', 1)
        name, count = description.rsplit(':', 1)
        cores = int(count)
        if not prefix or not name.strip() or cores <= 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError('Expected PREFIX=NAME:CORES with a positive integer core count') from None
    return prefix, name.strip(), cores


def node_type(text, processors):
    matches = [(name, cores) for prefix, name, cores in processors
               if re.search(r'\b' + re.escape(prefix), text)]
    if not matches:
        raise ValueError('Processor architecture not identified by the supplied --processor mappings')
    if len(matches) != 1:
        raise ValueError('Timing receipt matches more than one --processor mapping')
    return matches[0]


def matrix_times(specs, matrices, node_logs, processors):
    result = []
    for version, matrix in matrices.items():
        for spec in specs:
            run = run_path(matrix, version, spec['case'])
            timing = time_receipt(run / 'run.time.log')
            if timing is None:
                raise ValueError(f'Incomplete timing: {version}/{spec["case"]}')
            receipt = (run / 'START.txt').read_text()
            architecture, cores = node_type(receipt, processors)
            if node_logs:
                logs = list(node_logs.glob(f'*_e2e{version.replace(".", "")}f_{spec["case"]}.sh.log'))
                if not logs:
                    raise ValueError(f'Missing node log for {version}/{spec["case"]}')
                # Logs can contain an entire run; the host is in the short prologue.
                with logs[0].open() as h:
                    prologue = ''.join(next(h, '') for _ in range(12))
                host = re.search(r'host=([^ ]+)', receipt).group(1)
                if host not in prologue:
                    raise ValueError(f'Host receipt disagrees for {version}/{spec["case"]}')
            result.append(dict(case=spec['case'], design=spec['design'], variant=spec['variant'],
                               version=version, processor=architecture, node_cores=cores,
                               **timing, source=source(version, spec['case'], 'run.time.log;START.txt')))
    return result


def equal_cell(a, b):
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except (ValueError, TypeError):
        return False
    return (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-8)


def canonical_table(path, columns=None):
    records = rows(path)
    # Family numbers, sample row order and unordered parent slots are display conventions.
    family_ids = {}
    if path.parent.name == 'relatedness':
        members = defaultdict(list)
        for r in rows(path.parent / 'families.csv'):
            members[r['Family']].append(r['Sample'])
        family_ids = {f: ';'.join(sorted(names)) for f, names in members.items()}
    for r in records:
        r.pop('elapsed_seconds', None)
        r.pop('parent_slot', None)
        for key in ['Family', 'FamilyA', 'FamilyB']:
            if key in r:
                r[key] = family_ids[r[key]]
        for a, b in [('SampleA', 'SampleB'), ('FamilyA', 'FamilyB')]:
            if a in r and r[a] > r[b]:
                r[a], r[b] = r[b], r[a]
        if 'Parent1' in r and (not r['Parent1'] or (r['Parent2'] and r['Parent1'] > r['Parent2'])):
            for a, b in [('Parent1', 'Parent2'), ('Parent1Support', 'Parent2Support'),
                         ('Parent1Supported', 'Parent2Supported')]:
                if a in r:
                    r[a], r[b] = r[b], r[a]
        for key in ['parents', 'children']:
            if key in r:
                r[key] = ';'.join(sorted(r[key].split(';')))
    if path.name == 'sample_segments.tsv':
        groups = defaultdict(list)
        for r in records:
            groups[(r['SampleID'], r['Component'])].append(r)
        for group in groups.values():
            tracks = {t: sorted([(int(r['Start']), int(r['End']), r['Status'], r['Label'], r['Rows'])
                                  for r in group if r['Track'] == t]) for t in ['0', '1']}
            if tracks['1'] < tracks['0']:
                for r in group:
                    r['Track'] = str(1 - int(r['Track']))
    if columns:
        records = [{k: r[k] for k in columns} for r in records]
    return sorted(records, key=lambda r: tuple(r.values()))


def table_equal(a, b, columns=None):
    aa, bb = canonical_table(a, columns), canonical_table(b, columns)
    if len(aa) != len(bb):
        return False
    if not aa:
        return a.read_text().splitlines()[0] == b.read_text().splitlines()[0]
    return all(x.keys() == y.keys() and all(equal_cell(x[k], y[k]) for k in x)
               for x, y in zip(aa, bb))


def vcf_header(path):
    with gzip.open(path, 'rb') as handle:
        for line in handle:
            if line.startswith(b'#CHROM'):
                return line
    raise ValueError(f'Missing VCF header: {path.name}')


def vcf_digest(path, target_header=None):
    digest = hashlib.sha256()
    with gzip.open(path, 'rb') as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f'Empty VCF: {path.name}')
            if not line.startswith(b'##'):
                header = line
                break
        if target_header is None or target_header == header:
            digest.update(header)
            while block := handle.read(1024 * 1024):
                digest.update(block)
        else:
            original = header.rstrip().split(b'\t')
            target = target_header.rstrip().split(b'\t')
            if original[:9] != target[:9] or set(original[9:]) != set(target[9:]):
                return 'different-column-identities'
            order = [original.index(name) for name in target[9:]]
            digest.update(target_header)
            for line in handle:
                fields = line.rstrip(b'\n').split(b'\t')
                digest.update(b'\t'.join(fields[:9] + [fields[i] for i in order]) + b'\n')
    return digest.hexdigest()


def release_segments(path):
    return sorted((chrom['contig'], c['first_position'], c['last_position'],
                   len(c['source_founder_rows']), c['withheld_ambiguous_alleles'],
                   c['withheld_unobserved_alleles'])
                  for chrom in read_json(path)['contigs'] for c in chrom['founder_components'])


def robustness(specs, matrices):
    """Direct within-release comparisons; no cross-release inference of stability."""
    results = []
    scope = {
        'release_segments': ('export/manifest.json', ['contig', 'first_position', 'last_position', 'founder_row_count', 'withheld_ambiguous_alleles', 'withheld_unobserved_alleles']),
        'founder_counts': ('painting/*/founder_counts.tsv',
                           ['Component', 'FirstVariant1Based', 'LastVariant1Based', 'FounderRows', 'FounderCountAmbiguous']),
        'paintings': ('painting/*/sample_segments.tsv', None),
        'pedigree': ('pedigree/tier_*relationships.csv', None),
        'relatedness': ('relatedness/*.csv', None),
        'phase_summary': ('family_phase/summary.csv', ['contig', 'samples', 'sites', 'components', 'called_alleles', 'phased_heterozygotes', 'unphased_heterozygotes', 'phase_sets']),
        'recombination': ('recombination/*.csv', None),
        'founder_alleles': ('export/*.founders.vcf.gz', None),
        'sample_GT_PS': ('export/*.samples.vcf.gz', None),
    }
    digest_cache = {}
    for version, matrix in matrices.items():
        for design in ['tromau', 'actm3', 'std_n80', 'stress_n80']:
            group = [s for s in specs if s['design'] == design]
            ref = next(s for s in group if s['variant'] == 'ref')
            base = run_path(matrix, version, ref['case'])
            # 4.1 provides additional context for the known convention sensitivity.
            # Cross-version scores already compare ALL cases; within-release table is explicit.
            for spec in group:
                if spec['variant'] == 'ref':
                    continue
                run = run_path(matrix, version, spec['case'])
                print(f'Agreement {version} {spec["case"]}', flush=True)
                for family, (pattern, columns) in scope.items():
                    base_files = {str(p.relative_to(base)) for p in base.glob(pattern)}
                    run_files = {str(p.relative_to(run)) for p in run.glob(pattern)}
                    if not base_files or not run_files:
                        raise ValueError(f'Missing agreement products: {version}/{design}/{family}')
                    differences = sorted(base_files ^ run_files)
                    for name in sorted(base_files & run_files):
                        a, b = base / name, run / name
                        if family == 'release_segments':
                            same = release_segments(a) == release_segments(b)
                        elif name.endswith('.vcf.gz'):
                            for p in (a, b):
                                if p not in digest_cache:
                                    digest_cache[p] = vcf_digest(p, vcf_header(a))
                            same = digest_cache[a] == digest_cache[b]
                        else:
                            same = table_equal(a, b, columns)
                        if not same:
                            differences.append(name)
                    results.append(dict(version=version, design=design, variant=spec['variant'],
                                        case=spec['case'], reference_case=ref['case'], output_family=family,
                                        identical=int(not differences), compared_files=len(base_files & run_files),
                                        differing_files=';'.join(differences),
                                        source=f'{source(version, spec["case"], pattern)} vs {source(version, ref["case"], pattern)}'))
    return results, {k: {'pattern': v[0], 'columns': v[1] or 'all except elapsed_seconds and parent_slot; rows aligned; family IDs and unordered parent slots canonicalized'} for k, v in scope.items()}


def tromau_data(matrix, matrix41, out, mosaic_chr):
    run = run_path(matrix, '4.2', 'tromau')
    prefix = lambda name: source('4.2', 'tromau', name)
    families = rows(run / 'relatedness/families.csv')
    pedigree = {r['Sample']: r for r in rows(run / 'pedigree/tier_b_relationships.csv')}
    diagnostics = {r['Sample']: r for r in rows(run / 'pedigree/diagnostics.csv')}
    members = defaultdict(list)
    for row in families:
        members[row['Family']].append(row['Sample'])
    # Order only by inferred family size/membership, not by biological/cohort labels.
    family_order = sorted(members, key=lambda f: (-len(members[f]), sorted(members[f])))
    sample_order = [s for f in family_order for s in sorted(members[f])]
    people = []
    for f in family_order:
        for sample in sorted(members[f]):
            p, d = pedigree[sample], diagnostics[sample]
            r = next(r for r in families if r['Sample'] == sample)
            people.append(dict(sample=sample, plot_order=sample_order.index(sample), family=f,
                               family_size=len(members[f]), family_support=r['FamilySupport'],
                               parent1=p['Parent1'], parent2=p['Parent2'], parent_state=p['ParentState'],
                               inference_status=p['InferenceStatus'], release_evidence=p['ReleaseEvidence'],
                               configuration_bootstrap=d['SelectedGraphConfigurationBootstrapFraction'],
                               configuration_loco=d['SelectedGraphConfigurationLOCOFraction'],
                               parent1_bootstrap=d['Parent1BootstrapFraction'], parent2_bootstrap=d['Parent2BootstrapFraction'],
                               source=prefix('relatedness/families.csv;pedigree/{tier_b_relationships,diagnostics}.csv')))
    write_csv(out, 'tromau_samples.csv', people)
    mosaics = []
    for r in rows(run / f'painting/{mosaic_chr}/sample_segments.tsv'):
        mosaics.append(dict(chromosome=mosaic_chr, sample=r['SampleID'], component=r['Component'],
                            track=r['Track'], start_bp=r['Start'], end_bp=r['End'], status=r['Status'],
                            label=r['Label'], founder_rows=r['Rows'],
                            source=prefix(f'painting/{mosaic_chr}/sample_segments.tsv')))
    write_csv(out, 'tromau_mosaic.csv', mosaics)
    maps, meioses, founders = [], [], []
    for p in sorted((run / 'recombination').glob('*.map.csv'), key=lambda p: natural(p.name)):
        chrom = p.name.removesuffix('.map.csv')
        for r in rows(p):
            maps.append(dict(chromosome=chrom, **r, source=prefix('recombination/' + p.name)))
        for r in rows(run / f'recombination/{chrom}.meioses.csv'):
            meioses.append(dict(chromosome=chrom, **r, source=prefix(f'recombination/{chrom}.meioses.csv')))
    for version, case in [('4.1', 'tromau'), ('4.1', 'zero_tromau_rev'), ('4.2', 'tromau'), ('4.2', 'zero_tromau_rev')]:
        matrix_v = matrix if version == '4.2' else matrix41
        root = run_path(matrix_v, version, case)
        manifest = read_json(root / 'export/manifest.json')
        for chromosome in sorted(manifest['contigs'], key=lambda c: natural(c['contig'])):
            chrom = chromosome['contig']
            assembly = {r['Component']: r for r in rows(root / f'painting/{chrom}/founder_counts.tsv')}
            for c in chromosome['founder_components']:
                original = assembly[str(c['source_component'])]
                founders.append(dict(version=version, case=case, chromosome=chrom,
                                     component=c['component'], source_component=c['source_component'],
                                     start_bp=c['first_position'], end_bp=c['last_position'],
                                     founder_rows=len(c['source_founder_rows']),
                                     assembly_founder_rows=original['FounderRows'],
                                     ambiguous=original['FounderCountAmbiguous'],
                                     withheld_ambiguous_alleles=c['withheld_ambiguous_alleles'],
                                     source=source(version, case, f'export/manifest.json;painting/{chrom}/founder_counts.tsv')))
    write_csv(out, 'tromau_recombination.csv', maps)
    write_csv(out, 'tromau_meioses.csv', meioses)
    write_csv(out, 'tromau_founder_counts.csv', founders)



def tromau_processor_costs(matrix42, probes, node_logs, processors):
    """Read both completed 4.2 TroMau run/init receipts; retain processor provenance."""
    result = []
    experiments = [
        (run_path(matrix42, '4.2', 'tromau'),
         'matrix42/runs/e2e42f_tromau', '*_e2e42f_tromau.sh.log'),
        (probes / 'tromau_sapphire/run',
         'probes/tromau_sapphire/run', '*tromau_sapphire*.sh.log'),
    ]
    for root, logical_root, pattern in experiments:
        log = next(iter(sorted(node_logs.glob(pattern))))
        with log.open() as handle:
            receipt = ''.join(next(handle, '') for _ in range(12))
        architecture, cores = node_type(receipt, processors)
        recorded_cores = int(re.search(r'\bcores[= ](\d+)', receipt).group(1))
        if recorded_cores != cores:
            raise ValueError('TroMau timing does not use the stated full-node core count')
        if (root / 'START.txt').exists():
            host = re.search(r'host=([^ ]+)', (root / 'START.txt').read_text()).group(1)
            if host not in receipt:
                raise ValueError('TroMau matrix start receipt disagrees with node log')
        run = time_receipt(root / 'run.time.log')
        init = time_receipt(root / 'init.time.log')
        if run is None or init is None:
            raise ValueError('Incomplete TroMau processor comparison timing')
        sample_count = len(rows(root / 'relatedness/families.csv'))
        measurements = [
            ('wall_seconds', run['wall_seconds'], 'run.time.log'),
            ('peak_rss_kib', run['peak_rss_kib'], 'run.time.log'),
            ('init_wall_seconds', init['wall_seconds'], 'init.time.log'),
            ('init_peak_rss_kib', init['peak_rss_kib'], 'init.time.log'),
            ('init_plus_run_wall_seconds', round(init['wall_seconds'] + run['wall_seconds'], 6),
             'init.time.log;run.time.log'),
        ]
        for measurement, value, names in measurements:
            sources = ';'.join(f'{logical_root}/{name}' for name in names.split(';'))
            result.append(dict(experiment='TroMau_processor', version='4.2',
                               processor=architecture, cores=cores, samples=sample_count,
                               measurement=measurement, value=value,
                               source=f'{sources};node-logs/{pattern}'))
    return result


def cost_context():
    return dict(manuscript_target_release='4.4.0',
                tromau_processor_comparison=dict(
                    measured_version='4.2', plotted_time='run command only',
                    additional_retained_times='init and init plus run',
                    design='Full nodes of different processor architectures and core counts; one observation each',
                    released_result_agreement='Identical, as reported by the coordinator and owner; invocation timings excluded',
                    agreement_source='STATUS_42.md (completed TroMau Sapphire comparison)'))


def paired_costs(probes, node_logs, matrix42, processors):
    result = []
    for version, suffix in [('4.1', '41'), ('4.2', '42')]:
        run = probes / 'k30pair' / f'r{suffix}'
        t = time_receipt(run / 'run.time.log')
        sample_count = read_json(run / 'evaluation/summary.json')['samples']
        if t is None:
            raise ValueError('Incomplete paired K = 30 receipt')
        log = next(node_logs.glob(f'*k30pair_{suffix}.sh.log'))
        with log.open() as handle:
            receipt = ''.join(next(handle, '') for _ in range(3))
        architecture, cores = node_type(receipt, processors)
        result.append(dict(experiment='K30_whole_run', version=version, processor=architecture, cores=cores,
                           samples=sample_count, measurement='wall_seconds', value=t['wall_seconds'],
                           source=f'probes/k30pair/r{suffix}/run.time.log;node-logs/*k30pair_{suffix}.sh.log'))
        result.append(dict(experiment='K30_whole_run', version=version, processor=architecture, cores=cores,
                           samples=sample_count, measurement='peak_rss_kib', value=t['peak_rss_kib'],
                           source=f'probes/k30pair/r{suffix}/run.time.log'))
        timing = read_json(probes / 'ped2500' / f'timing_{suffix}.json')
        architecture, _ = node_type(timing['host'], processors)
        log_text = (probes / 'ped2500' / f'log_{suffix}.txt').read_text()
        pedigree_n = int(re.search(r'fixture .*: N=(\d+),', log_text).group(1))
        for measurement in ['decision_seconds', 'seam_seconds', 'total_seconds']:
            result.append(dict(experiment='N2503_pedigree', version=version, processor=architecture,
                               cores=timing['cores'], samples=pedigree_n, measurement=measurement,
                               value=timing[measurement], source=f'probes/ped2500/timing_{suffix}.json;probes/ped2500/log_{suffix}.txt'))
        t = time_receipt(probes / 'ped2500' / f'time_{suffix}.txt')
        for measurement in ['wall_seconds', 'peak_rss_kib']:
            result.append(dict(experiment='N2503_pedigree', version=version, processor=architecture,
                               cores=timing['cores'], samples=pedigree_n, measurement=measurement,
                               value=t[measurement], source=f'probes/ped2500/time_{suffix}.txt'))
    result.extend(tromau_processor_costs(matrix42, probes, node_logs, processors))
    return result


def canonical_table44(path, columns):
    """Align the snake_case 4.3/4.4 layout without rematching founder identities."""
    records = rows(path)
    family_ids = {}
    if path.parent.name == 'relatedness':
        members = defaultdict(list)
        for r in rows(path.parent / 'families.csv'):
            members[r['family']].append(r['sample'])
        family_ids = {key: ';'.join(sorted(value)) for key, value in members.items()}
    for r in records:
        r.pop('elapsed_seconds', None)
        r.pop('parent_slot', None)
        for key in ['family', 'family_a', 'family_b']:
            if key in r:
                r[key] = family_ids[r[key]]
        for a, b in [('sample_a', 'sample_b'), ('family_a', 'family_b')]:
            if a in r and r[a] > r[b]:
                r[a], r[b] = r[b], r[a]
        if 'parent1' in r and (not r['parent1'] or (r['parent2'] and r['parent1'] > r['parent2'])):
            for a in list(r):
                if a.startswith('parent1'):
                    b = a.replace('parent1', 'parent2', 1)
                    if b in r:
                        r[a], r[b] = r[b], r[a]
        for key in ['parents', 'children']:
            if key in r:
                r[key] = ';'.join(sorted(r[key].split(';')))
    if path.name == 'sample_segments.csv':
        groups = defaultdict(list)
        for r in records:
            groups[(r['contig'], r['sample'], r['component'])].append(r)
        for group in groups.values():
            tracks = {t: sorted((int(r['start']), int(r['end']), r['status'], r['label'], r['rows'])
                                for r in group if r['track'] == t) for t in ['0', '1']}
            if tracks['1'] < tracks['0']:
                for r in group:
                    r['track'] = str(1 - int(r['track']))
    if columns:
        records = [{key: r[key] for key in columns} for r in records]
    return sorted(records, key=lambda r: tuple(r.values()))


def robustness44(matrix, out):
    """Direct selected-product agreement within 4.4; no real-data truth claim."""
    scope = {
 'release_segments':('export/manifest.json',None),
 'founder_counts':('painting/founder_counts.csv',['contig','component','first_variant_1based','last_variant_1based','founder_rows','founder_count_ambiguous']),
 'paintings':('painting/sample_segments.csv',None),
 'pedigree':('pedigree/tier_*relationships.csv',None),
 'relatedness':('relatedness/*.csv',None),
 'phase_summary':('family_phase/summary.csv',['contig','samples','sites','components','called_alleles','phased_heterozygotes','unphased_heterozygotes','phase_sets']),
 'recombination':('recombination/*.csv',None),
 'founder_alleles':('export/*.founders.vcf.gz',None),
 'sample_GT_PS':('export/*.samples.vcf.gz',None)}
    digest_cache, results = {}, []
    for design, reference in [('tromau', 'tromau'), ('actm3', 'zero_actm3_ref')]:
        base = matrix / 'runs' / ('e2e44_' + reference)
        for variant in ['iid12_s1', 'iid9_s1', 'val12_s1', 'perm_s1', 'vcfperm_s1', 'rev']:
            case = 'zero_' + design + '_' + variant
            run = base.parent / ('e2e44_' + case)
            for family, (pattern, columns) in scope.items():
                aa = {str(p.relative_to(base)): p for p in base.glob(pattern)}
                bb = {str(p.relative_to(run)): p for p in run.glob(pattern)}
                if not aa or not bb:
                    raise ValueError('Missing 4.4 agreement products')
                differences = sorted(aa.keys() ^ bb.keys())
                for name in sorted(aa.keys() & bb.keys()):
                    a, b = aa[name], bb[name]
                    if family == 'release_segments':
                        same = release_segments(a) == release_segments(b)
                    elif name.endswith('.vcf.gz'):
                        header = vcf_header(a)
                        for p in (a, b):
                            key = (p, header)
                            if key not in digest_cache:
                                digest_cache[key] = vcf_digest(p, header)
                        same = digest_cache[(a, header)] == digest_cache[(b, header)]
                    else:
                        x, y = canonical_table44(a, columns), canonical_table44(b, columns)
                        same = len(x) == len(y) and all(
                            u.keys() == v.keys() and all(equal_cell(u[k], v[k]) for k in u)
                            for u, v in zip(x, y))
                        if not x and not y:
                            same = a.read_text().splitlines()[0] == b.read_text().splitlines()[0]
                    if not same:
                        differences.append(name)
                results.append(dict(version='4.4.0', design=design, variant=variant, family=family,
                                    files=len(aa), equal=int(not differences), differing_files=';'.join(differences),
                                    source='matrix44/runs/{e2e44_' + reference + ',e2e44_' + case + '}/' + pattern))
            print(f'Agreement 4.4 {case}', flush=True)
    write_csv(out, 'robustness44.csv', results)
    return dict(source='matrix44/runs/e2e44_*/',
                scope={k: dict(pattern=v[0], columns=v[1]) for k, v in scope.items()},
                numeric_table_relative_tolerance=1e-9, numeric_table_absolute_tolerance=1e-8,
                vcf='Exact SHA-256 of aligned #CHROM header and records, excluding ## metadata; founder labels, missingness, GT and PS retained',
                alignment='Sample IDs; families by membership; unordered parents; undirected endpoints; whole painting-track exchange per sample/component; elapsed_seconds and parent_slot excluded',
                summary='Both real datasets match in all nine selected families under iid12, iid9, val12, perm and vcfperm; actm3 also under reversal. TroMau reversal changes seven families.',
                limitations=['No real-data truth or cross-release identity verdict.',
                             'Exact raw comparisons in depth-comparison/realdata/*.json include numerical diagnostic differences and sample order; their full-product passed flags are false.',
                             'Selected founder-count and phase-summary columns exclude search and floating objective diagnostics.'])


def depth_argument(value):
    try:
        depth, path = value.split('=', 1)
        depth = float(depth)
        if not math.isfinite(depth) or depth <= 0 or not path:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError('Expected positive DEPTH=PATH') from None
    return depth, Path(path)


def final_truth(summary):
    """Pool founder counts using the manuscript's matched-call denominator."""
    founders = [r for r in summary['founders'] if r['stage'] == 'founder_refit']
    if not founders:
        raise ValueError('Final founder-refit evaluation required')
    total = lambda key: sum(int(r[key]) for r in founders)
    t, p = summary['totals'], summary['pedigree']
    for key in ['called_genotypes', 'genotype_errors', 'switch_errors', 'set_aligned_errors',
                'phased_true_heterozygotes', 'correct_true_heterozygotes']:
        if t[key] != sum(r[key] for r in summary['chromosomes']):
            raise ValueError(f'Chromosome totals disagree for {key}')
    return dict(samples=summary['samples'], chromosomes=len(summary['chromosomes']), seed=summary['seed'],
                founder_stage='founder_refit', founder_called_alleles=total('called_alleles'),
                founder_matched_called_alleles=total('matched_called_alleles'),
                founder_wrong=total('one_to_one_called_allele_errors'),
                founder_missing=total('missing_alleles'), founder_truth_alleles=total('truth_alleles'),
                founder_wrong_ppm=ratio(1e6 * total('one_to_one_called_allele_errors'), total('matched_called_alleles')),
                founder_missing_ppm=ratio(1e6 * total('missing_alleles'), total('called_alleles') + total('missing_alleles')),
                called_genotypes=t['called_genotypes'], genotype_errors=t['genotype_errors'],
                genotype_errors_ppm=ratio(1e6 * t['genotype_errors'], t['called_genotypes']),
                phased_true_heterozygotes=t['phased_true_heterozygotes'],
                correct_true_heterozygotes=t['correct_true_heterozygotes'],
                phased_share=ratio(t['phased_true_heterozygotes'], t['correct_true_heterozygotes']),
                switch_errors=t['switch_errors'], long_switches=t['long_switches'],
                set_aligned_errors=t['set_aligned_errors'],
                set_errors_ppm=ratio(1e6 * t['set_aligned_errors'], t['phased_true_heterozygotes']),
                released_phase_sets=t['released_phase_sets'],
                released_phase_set_n50_bp=t['released_phase_set_n50_bp'],
                true_edges=p['true_edges'], correct_edges=p['correct_edges'],
                missing_edges=p['missing_edges'], extra_edges=p['extra_edges'],
                correct_parent_states=p['correct_parent_states'], exact_configurations=p['exact_configurations'])


def evaluation_root(root):
    for path in [root / 'run/evaluation', root / 'evaluation', root]:
        if (path / 'summary.json').is_file():
            return path
    raise ValueError('No evaluation summary in supplied depth source')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation_comparison(a, b):
    """Exact evaluation values, allowing numeric CSV spelling and checkpoint roots."""
    aa = {p.name: p for p in a.iterdir() if p.suffix in ('.csv', '.json')}
    bb = {p.name: p for p in b.iterdir() if p.suffix in ('.csv', '.json')}
    differences = sorted(aa.keys() ^ bb.keys())
    for name in sorted(aa.keys() & bb.keys()):
        if name.endswith('.json'):
            x, y = read_json(aa[name]), read_json(bb[name])
            x.pop('checkpoint_root', None)
            y.pop('checkpoint_root', None)
            same = x == y
        else:
            x, y = rows(aa[name]), rows(bb[name])
            def exact(u, v):
                if u == v:
                    return True
                try:
                    from decimal import Decimal
                    a, b = Decimal(u), Decimal(v)
                    return (a.is_nan() and b.is_nan()) or a == b
                except (ValueError, ArithmeticError):
                    return False
            same = len(x) == len(y) and all(
                u.keys() == v.keys() and all(exact(u[k], v[k]) for k in u) for u, v in zip(x, y))
        if not same:
            differences.append(name)
    return dict(files=len(aa), equal=not differences, differing_files=differences,
                scope='All evaluation CSV/JSON values and order; checkpoint_root excluded; numeric CSV spelling equivalent; no tolerance')


def depth_data(args, metadata):
    depths = dict(args.depth_run)
    if len(depths) != len(args.depth_run):
        raise ValueError('Duplicate --depth-run depth')
    baselines = dict(args.depth_baseline43 or [])
    if len(baselines) != len(args.depth_baseline43 or []):
        raise ValueError('Duplicate --depth-baseline43 depth')
    if baselines.keys() - depths.keys():
        raise ValueError('Every baseline depth requires a corresponding --depth-run')
    records, receipts, provenance = [], [], {}
    for depth, base in sorted(depths.items()):
        label = f'depth44/d{depth:g}'
        eroot = evaluation_root(base)
        summary = read_json(eroot / 'summary.json')
        config = tomllib.loads((base / 'config.toml').read_text())
        if config['simulation']['depth'] != depth:
            raise ValueError('Depth argument disagrees with config')
        if not re.search(r'\brc=0\s+eval_rc=0\b', (base / 'RESULT.txt').read_text()):
            raise ValueError('Depth run did not complete successfully')
        commands = [read_json(p) for p in sorted((base / 'run/run_records').glob('*.json'))]
        run = [r for r in commands if r['command'] == 'run']
        if len(run) != 1 or any(r['state'] != 'complete' for r in commands):
            raise ValueError('Depth series requires one completed run attempt')
        run = run[0]
        architecture, cores = node_type(run['host'], args.processor)
        if run['cpu_affinity'] != cores:
            raise ValueError('Depth run does not use the stated full-node affinity')
        timing = time_receipt(base / 'run.time.log')
        if timing is None:
            raise ValueError('Missing successful depth run timing')
        memory = run['memory']
        last = [line for line in (base / 'run.log').read_text().splitlines() if '[memory]' in line][-1]
        admission = re.search(r'(\d+) admissions, (\d+) waits, (\d+) forced grants', last)
        if not admission or tuple(map(int, admission.groups())) != tuple(memory[k] for k in ('admissions', 'waits', 'forced')):
            raise ValueError('Memory log and run record disagree')
        counts = final_truth(summary)
        records.append(dict(depth=depth, version='4.4.0', **counts,
                            processor=architecture, node_cores=cores,
                            run_wall_seconds=timing['wall_seconds'], run_peak_rss_kib=timing['peak_rss_kib'],
                            peak_live_bytes=memory['peak_live_use_bytes'], memory_limit_bytes=memory['limit_bytes'],
                            configured_limit_bytes=memory['configured_bytes'] or '',
                            admissions=memory['admissions'], waits=memory['waits'], forced_grants=memory['forced'],
                            source=label + '/{config.toml,run.time.log,run.log,run/evaluation/summary.json,run/run_records/*.json}'))
        provenance[f'{depth:g}'] = dict(source=label, config_sha256=file_hash(base / 'config.toml'),
                                       evaluation_sha256=file_hash(eroot / 'summary.json'),
                                       run_time_sha256=file_hash(base / 'run.time.log'))
        if depth in baselines:
            old = evaluation_root(baselines[depth])
            comparison = evaluation_comparison(old, eroot)
            comparison['source'] = f'depth43/d{depth:g}/evaluation'
            provenance[f'{depth:g}']['baseline43'] = comparison
            if not comparison['equal']:
                raise ValueError(f'4.3/4.4 evaluation differences at depth {depth:g}: {comparison["differing_files"]}')
    # Keep old attempts separate: a successful resumed command is not a cold whole run.
    for depth, base in sorted(baselines.items()):
        if not (base / 'run/run_records').is_dir():
            continue
        attempt = 0
        for p in sorted((base / 'run/run_records').glob('*.json')):
            r = read_json(p)
            if r['command'] != 'run':
                continue
            attempt += 1
            architecture, cores = node_type(r['host'], args.processor)
            m = r['memory']
            receipts.append(dict(depth=depth, version='4.3.0', attempt=attempt, state=r['state'],
                                 processor=architecture, node_cores=cores,
                                 runtime_wall_seconds=r['wall_seconds'],
                                 peak_live_bytes=m['peak_live_use_bytes'], memory_limit_bytes=m['limit_bytes'],
                                 waits=m['waits'], forced_grants=m['forced'],
                                 source=f'depth43/d{depth:g}/run/run_records/{p.name}'))
    write_csv(args.out, 'depth_series.csv', records)
    if receipts:
        write_csv(args.out, 'depth_baselines.csv', receipts)
    metadata['depth_series44'] = dict(
        release='4.4.0', commit='867f163', replicates_per_depth=1, sources=provenance,
        definitions=dict(founder_wrong_ppm='One-to-one errors / matched called founder alleles * 1e6; all rows matched in this series, so identical to compare/metrics.py all-call denominator',
                         founder_missing_ppm='Missing founder alleles / (called + missing founder alleles) * 1e6',
                         genotype_errors_ppm='Genotype errors / called genotypes * 1e6',
                         set_errors_ppm='Minority-orientation sites within each released set / phased true heterozygotes * 1e6',
                         phased_share='Phased true heterozygotes / correctly called true heterozygotes',
                         released_phase_set_n50_bp='Pooled released-set span N50, not mean chromosome N50',
                         peak_live_bytes='Run record sampled memory pressure; includes concurrent work and node pressure; not summed RSS',
                         run_peak_rss_kib='GNU time maximum process RSS of run command only; separate init/evaluate excluded',
                         run_wall_seconds='GNU time elapsed run command, one successful attempt; runtime timer excludes some command overhead'),
        limitations=['One replicate per depth; no uncertainty across simulated datasets.',
                     'Mixed processor architectures, CPU counts and concurrent lane counts; no cross-node speedup or depth-only memory effect.',
                     'Depths 1 and 1.5 use configured 250 GiB governor limits on larger nodes.',
                     'Limits are recorded bytes, not nominal node labels.'])


def matrix44_provenance(root, out, metadata):
    """Recheck retained scalar evidence without running the matrix's writing tools."""
    reference = {(r['case'], r['version']): r for r in rows(out / 'simulation_accuracy.csv')}
    baseline = {(r['case'], r['version']): r for r in rows(root / 'baseline43/simulation_accuracy.csv')}
    results, fields = [], None
    for p in sorted((root / 'scores').glob('*.json')):
        if p.name == 'summary.json':
            continue
        r = read_json(p)
        old = read_json(root / 'baseline43' / p.name)
        if not old['passed'] or old['counts']['scientific_differences']:
            raise ValueError('Archived baseline does not establish 4.3/4.2 equality')
        commands = ['init', 'run', 'eval'] if r['status'] == 'PASS_PROJECTED_TRUTH' else ['init', 'run']
        execution = read_json(root / 'runs' / ('e2e44_' + r['case']) / 'RESULT.json')
        if any(execution[k] != 0 for k in ['rc', 'eval_rc', 'compare_rc']) or not all(r['performance44'][k + '_complete'] for k in commands):
            raise ValueError('Incomplete matrix case')
        if r['status'] == 'PASS_PROJECTED_TRUTH':
            current_fields = [c['metric'] for c in r['comparisons']]
            fields = current_fields if fields is None else fields
            if fields != current_fields or not all(c['equal'] for c in r['comparisons']):
                raise ValueError('Matrix scalar comparisons disagree')
            key = (r['case'], '4.2')
            if reference[key] != baseline[key]:
                raise ValueError('Tracked scalar summary differs from archived baseline')
            for c in r['comparisons']:
                if not equal_cell(reference[key][c['metric']], c['actual']):
                    raise ValueError('Matrix score differs from tracked scalar summary')
        elif r['status'] != 'NO_TRUTH':
            raise ValueError('Unexpected matrix status')
        results.append(dict(case=r['case'], status=r['status']))
    metadata['release44_validation'] = dict(
        source='matrix44/{scores,baseline43}/*.json', cases=len(results),
        simulation_cases=sum(r['status'] == 'PASS_PROJECTED_TRUTH' for r in results),
        real_data_cases=sum(r['status'] == 'NO_TRUTH' for r in results),
        matched_fields=fields, results=results,
        baseline='Retained 4.2 scalar truth fields carried to 4.3 by each case archived zero-scientific-difference comparison; checked against unchanged data/simulation_accuracy.csv',
        limitations=['Full 4.3 matrix evaluations and raw outputs were deleted; only retained scalar fields checked transitively.',
                     'Founder component, continuity, phase-stratum and recombination truth fields absent from the scalar summary are not newly established equal.',
                     'Real-data cases have no individual truth; within-4.4 released-product tests are separate.'],
        noise_hook='Declared compact-evidence materialisation before unchanged frozen noise draws; matrix44/HOOKS.md and prepare.py')
    metadata['manuscript_target_release'] = '4.4.0'
    metadata['release43_validation'] = 'Archived 39-case comparisons passed against mapped 4.2 outputs; raw 4.3 matrix outputs no longer retained'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix42', type=Path)
    parser.add_argument('--matrix41', type=Path)
    parser.add_argument('--probes', type=Path)
    parser.add_argument('--node-logs', type=Path, help='Node queue log directory for processor identification and receipt checks')
    parser.add_argument('--processor', action='append', type=processor_argument, required=True,
                        metavar='PREFIX=NAME:CORES', help='Repeat for every host prefix in the receipts; no built-in mappings')
    parser.add_argument('--out', type=Path, default=PAPER / 'data')
    parser.add_argument('--mosaic-chromosome', default='chr1')
    parser.add_argument('--costs-only', action='store_true', help='Refresh paired costs and cost provenance in an existing summary directory')
    parser.add_argument('--depth-run', action='append', type=depth_argument, metavar='DEPTH=PATH')
    parser.add_argument('--depth-baseline43', action='append', type=depth_argument, metavar='DEPTH=PATH')
    parser.add_argument('--realdata44', action='store_true', help='Compare selected released products within --matrix44, including streamed VCFs')
    parser.add_argument('--depth-comparison43', type=Path, help='Saved exact depth-2 cross-release comparison JSON')
    parser.add_argument('--matrix44', type=Path, help='Audit saved scores and baseline without running matrix tools')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.realdata44 and not args.matrix44:
        parser.error('--realdata44 requires --matrix44')
    if args.depth_baseline43 and not args.depth_run:
        parser.error('--depth-baseline43 requires --depth-run')
    if args.depth_run or args.matrix44:
        if args.costs_only:
            parser.error('--costs-only cannot be combined with depth/matrix44 extraction')
        metadata = read_json(args.out / 'provenance.json')
        if args.depth_run:
            depth_data(args, metadata)
        elif args.depth_baseline43:
            parser.error('--depth-baseline43 requires --depth-run')
        if args.matrix44:
            matrix44_provenance(args.matrix44, args.out, metadata)
        if args.realdata44:
            metadata['realdata44_validation'] = robustness44(args.matrix44, args.out)
        if args.depth_comparison43:
            comparison = read_json(args.depth_comparison43)
            if not comparison['passed'] or comparison['counts']['scientific_differences']:
                raise ValueError('Depth-2 exact output comparison failed')
            metadata['depth2_output_comparison43'] = dict(source='depth-comparison/d2_43_vs_44_final.json',
                                                         sha256=file_hash(args.depth_comparison43),
                                                         counts=comparison['counts'], policy=comparison['policy'])
        (args.out / 'provenance.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
        return
    if not all([args.matrix42, args.matrix41, args.probes, args.node_logs]):
        parser.error('Legacy extraction requires --matrix42, --matrix41, --probes and --node-logs; use depth/matrix44 mode to retain existing summaries')
    if args.costs_only:
        metadata = read_json(args.out / 'provenance.json')
        write_csv(args.out, 'paired_costs.csv', paired_costs(args.probes, args.node_logs, args.matrix42, args.processor))
        metadata.update(cost_context())
        metadata['limitations'] = [note for note in metadata['limitations']
                                   if not note.startswith('TroMau Sapphire probe excluded')]
        (args.out / 'provenance.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
        return
    specs = [s for s in read_json(args.matrix42 / 'final/matrix.json') if s['runnable']]
    specs.sort(key=lambda s: (DESIGNS.index(s['design']), VARIANTS.index(s['variant'])))
    matrices = {'4.1': args.matrix41, '4.2': args.matrix42}
    cases, accuracy, components, stages = [], [], [], {}
    for spec in specs:
        run = run_path(args.matrix42, '4.2', spec['case'])
        config = tomllib.loads((args.matrix42 / 'final/configs' / (spec['case'] + '.toml')).read_text())
        simulation = config['input']['kind'] == 'simulation'
        samples = read_json(run / 'evaluation/summary.json')['samples'] if simulation else len(rows(run / 'relatedness/families.csv'))
        sim = config.get('simulation', {}) if simulation else {}
        cases.append(dict(case=spec['case'], design=spec['design'], variant=spec['variant'],
                          input_kind=config['input']['kind'], has_truth=int(simulation), samples=samples,
                          chromosomes=len(config['input']['contigs']), contigs=';'.join(config['input']['contigs']),
                          depth=sim.get('depth', ''), seed=sim.get('seed', ''),
                          observed_fraction=sim.get('observed_fraction', ''),
                          depth_cv=sim.get('reads', {}).get('depth_cv', ''),
                          read_error_rate=sim.get('reads', {}).get('error_rate', ''),
                          dropout_fraction=sim.get('reads', {}).get('dropout_fraction', ''),
                          heterozygote_alt_probability=sim.get('reads', {}).get('heterozygote_alt_probability', ''),
                          backcross_fraction=sim.get('backcross_fraction', ''),
                          source=f'matrix42/final/configs/{spec["case"]}.toml'))
        if simulation:
            for version, matrix in matrices.items():
                a, c, st = evaluation(run_path(matrix, version, spec['case']), version, spec)
                accuracy.append(a)
                components.extend(c)
                stages[f'{version}/{spec["case"]}'] = st
    write_csv(args.out, 'cases.csv', cases)
    write_csv(args.out, 'simulation_accuracy.csv', accuracy)
    write_csv(args.out, 'founder_components.csv', components)
    write_csv(args.out, 'whole_run_times.csv', matrix_times(specs, matrices, args.node_logs, args.processor))
    write_csv(args.out, 'paired_costs.csv', paired_costs(args.probes, args.node_logs, args.matrix42, args.processor))
    tromau_data(args.matrix42, args.matrix41, args.out, args.mosaic_chromosome)
    agreement, scope = robustness(specs, matrices)
    write_csv(args.out, 'robustness.csv', agreement)
    # Cross-release score summaries provide the full diagnostic context without copying long logs.
    comparisons = []
    for spec in specs:
        score = read_json(args.matrix42 / 'scores42' / (spec['case'] + '.json'))
        comparisons.append(dict(case=spec['case'], compared_files=score['compared'],
                                differing_files=';'.join(sorted(score['differing'])),
                                only42=';'.join(score['only41']), only41=';'.join(score['only40']),
                                source=f'matrix42/scores42/{spec["case"]}.json'))
    write_csv(args.out, 'release_comparisons.csv', comparisons)
    metadata = dict(schema=1, releases=['4.1', '4.2'], cases=len(cases),
                    simulation_cases=sum(c['has_truth'] for c in cases), evaluation_stages=stages,
                    robustness_scope=scope, numeric_table_relative_tolerance=1e-9, numeric_table_absolute_tolerance=1e-8,
                    vcf_agreement='SHA-256 of #CHROM and all records; ## metadata excluded; sample columns aligned to reference; founder labels and phase orientation retained',
                    sources='Logical paths relative to supplied matrix/probe roots; no absolute paths or host names stored',
                    timing_scope='run command only, excluding separate init and evaluate; no repeated-run error bars',
                    peak_rss_scope='GNU time maximum resident set size; not a sum of worker RSS',
                    mosaic_chromosome=args.mosaic_chromosome,
                    limitations=['Only founder_refit truth scores are present; no stage-by-stage trajectory.',
                                 'Perturbations are repeated analyses of the same data, not independent simulations.',
                                 'TroMau and actm3 have no individual-level ground truth.',
                                 'Stress changes several conditions simultaneously; no depth-only sensitivity curve.'])
    metadata.update(cost_context())
    (args.out / 'provenance.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
