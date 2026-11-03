#!/usr/bin/env python3
"""Draw results figures 3–12 from compact paper42/data/ CSV/JSON only.

Writes figures/fig<N>_<name>.svg and .pdf using matplotlib (no run-directory
access, pipeline imports, or measurements). The CSVs are produced separately by
figure_data.py. Captions and source/denominator definitions are in CAPTIONS.md.

Run from anywhere: python -P paper42/tools/make_results_figures.py
Use --data-dir and --output-dir for an isolated regeneration check. SVG identifiers
and PDF metadata are fixed so identical inputs give byte-identical artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
# Okabe–Ito palette; state is also encoded by position, symbols or hatching.
BLUE, ORANGE, GREEN, PURPLE = '#0072B2', '#D55E00', '#009E73', '#CC79A7'
SKY, YELLOW, GOLD, INK = '#56B4E9', '#F0E442', '#E69F00', '#222222'
GREY, LIGHT = '#777777', '#E7E7E7'
FOUNDERS = [BLUE, GOLD, GREEN, PURPLE, SKY, ORANGE, YELLOW, INK]
DESIGNS = ['std', 'std_n80', 'stress_n80', 'backcross', 'bc8', 'longbc', 'f3',
           'f3x8', 'longf3', 'famhs', 'hsrand', 'sibrich', 'seed', 'tl', 'k30']
VARIANTS = ['ref', 'perm_s1', 'iid12_s1', 'iid9_s1', 'val12_s1', 'vcfperm_s1', 'rev']
VNAME = {'ref': 'reference', 'perm_s1': 'sample order', 'iid12_s1': 'noise 10⁻¹²',
         'iid9_s1': 'noise 10⁻⁹', 'val12_s1': 'validation subset',
         'vcfperm_s1': 'VCF order', 'rev': 'reverse ties'}
DNAME = {'std_n80': 'std n80', 'stress_n80': 'stress n80 ◆', 'k30': 'K = 30 ★'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9,
    'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .6, 'lines.linewidth': 1, 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'svg.hashsalt': 'paper42-results-v1',
    'savefig.facecolor': 'white', 'axes.titlelocation': 'left',
})


def table(data, name):
    with (data / name).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def number(value):
    return float(value) if value not in ('', None) else math.nan


def design_name(design):
    return DNAME.get(design, design)


def case_name(row):
    variant = row['variant']
    return design_name(row['design']) + ('' if variant == 'ref' else ' · ' + {
        'perm_s1': 'perm', 'iid12_s1': 'iid12', 'iid9_s1': 'iid9', 'val12_s1': 'val',
        'vcfperm_s1': 'VCF', 'rev': 'rev'}[variant])


def special(row):
    if row['design'] == 'k30':
        return ORANGE, '*', 44
    if row['design'] == 'stress_n80':
        return PURPLE, 'D', 15
    return BLUE, 'o', 12


def save(fig, out, stem, *, raster_dpi=None):
    for ext in ['svg', 'pdf']:
        metadata = {'Date': None, 'Creator': 'paper42/tools/make_results_figures.py'} if ext == 'svg' else {
            'CreationDate': None, 'ModDate': None, 'Creator': 'paper42/tools/make_results_figures.py'}
        target = out / f'{stem}.{ext}'
        options = {} if raster_dpi is None else {'dpi': raster_dpi}
        fig.savefig(target, metadata=metadata, bbox_inches='tight', pad_inches=.05, **options)
        if ext == 'svg':
            # Preserve the viewBox, but use integer CSS-pixel dimensions so the
            # existing manuscript browser/Word rasterization can read the SVG size.
            content = target.read_text()
            content = re.sub(r'width="([\d.]+)pt" height="([\d.]+)pt"',
                             lambda m: f'width="{math.ceil(float(m[1]) * 4 / 3)}" height="{math.ceil(float(m[2]) * 4 / 3)}"',
                             content, count=1)
            target.write_text(content)
    plt.close(fig)
    print(stem, flush=True)


def panel(ax, title):
    ax.set_title(title, pad=7, fontweight='bold')
    ax.tick_params(length=2.5, width=.6)


def rows_background(ax, records):
    groups = []
    for i, row in enumerate(records):
        if i == 0 or row['design'] != records[i - 1]['design']:
            groups.append(i)
    groups.append(len(records))
    for j, (a, b) in enumerate(zip(groups, groups[1:])):
        if j % 2 == 0:
            ax.axhspan(a - .5, b - .5, color='#F3F3F3', zorder=0)
    ax.set_ylim(len(records) - .4, -.6)
    ax.set_yticks(range(len(records)), [case_name(r) for r in records], fontsize=7)
    ax.grid(axis='x', color=LIGHT, linewidth=.5, zorder=0)
    ax.set_axisbelow(True)


def figure_accuracy(data, out):
    records = [r for r in table(data, 'simulation_accuracy.csv') if r['version'] == '4.2']
    records.sort(key=lambda r: (DESIGNS.index(r['design']), VARIANTS.index(r['variant'])))
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 9.7), sharey=True, layout='constrained')
    specs = [('founder_errors_per_million', 1, 'a  Founder errors', 'One-to-one errors / million\ncalled founder alleles', True),
             ('genotype_error_rate', 1e6, 'b  Genotype errors', 'Errors / million\ncalled genotypes', False),
             ('switches_per_phased_heterozygote', 1e6, 'c  Phase switches', 'Switches / million phased\ntrue heterozygotes', True),
             ('phased_share', 100, 'd  Phased share', 'Phased / correctly called\ntrue heterozygotes (%)', False),
             ('called_share', 100, 'e  Allele callability', 'Called sample alleles (%)', False)]
    for ax, (key, scale, title, label, symlog) in zip(axes.flat, specs):
        rows_background(ax, records)
        for i, r in enumerate(records):
            color, marker, size = special(r)
            ax.scatter(number(r[key]) * scale, i, color=color, marker=marker, s=size, zorder=3)
        if symlog:
            ax.set_xscale('symlog', linthresh=1)
            maximum = max(number(r[key]) * scale for r in records)
            ticks = [t for t in [0, 1, 10, 100, 1000, 10000, 100000] if t <= maximum * 1.3]
            ax.set_xticks(ticks, [f'{t:g}' for t in ticks])
            label += '\n(symlog; linear 0–1)'
            ax.set_xlim(0, maximum * 1.3)
        ax.set_xlim(left=0)
        if key in ['phased_share', 'called_share']:
            ax.set_xlim(0, 102)
        ax.set_xlabel(label, fontsize=7)
        panel(ax, title)
    ax = axes.flat[-1]
    rows_background(ax, records)
    for key, color, hatch, label in [('correct_edges', BLUE, '', 'Correct'),
                                    ('missed_edges', ORANGE, '//', 'Missed'),
                                    ('extra_edges', PURPLE, 'xx', 'Extra')]:
        left = np.array([0 if key == 'correct_edges' else int(r['correct_edges']) if key == 'missed_edges'
                         else int(r['correct_edges']) + int(r['missed_edges']) for r in records])
        ax.barh(range(len(records)), [int(r[key]) for r in records], left=left,
                height=.65, color=color, hatch=hatch, linewidth=0, label=label)
    for i, r in enumerate(records):
        if int(r['true_edges']) == 0:
            ax.text(3, i, 'no observed truth edges', va='center', fontsize=5.8, color=GREY)
    ax.set_xlim(left=0)
    ax.set_xlabel('Pedigree edges (count)')
    ax.legend(loc='lower right', frameon=False, fontsize=6)
    panel(ax, 'f  Pedigree edges')
    for i, a in enumerate(axes.flat):
        a.tick_params(labelleft=(i % 3 == 0))
    save(fig, out, 'fig3_simulation_accuracy')


def figure_founder_endpoint(data, out):
    records = [r for r in table(data, 'founder_components.csv')
               if r['version'] == '4.2' and r['variant'] == 'ref']
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.3), layout='constrained', gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axes[0]
    for i, design in enumerate(DESIGNS):
        group = [r for r in records if r['design'] == design]
        values = sorted(number(r['one_to_one_errors_per_million']) for r in group)
        if i % 2 == 0:
            ax.axhspan(i - .5, i + .5, color='#F3F3F3')
        for j, (r, value) in enumerate(sorted(zip(group, values), key=lambda x: x[1])):
            color, marker, size = special({'design': design})
            offset = (j - (len(values) - 1) / 2) * .065
            ax.scatter(value, i + offset, c=color, marker=marker, s=size, alpha=.8, zorder=3)
    ax.set_yticks(range(len(DESIGNS)), [design_name(d) for d in DESIGNS])
    ax.set_ylim(len(DESIGNS) - .4, -.6)
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=0)
    ax.set_xticks([0, 1, 100, 10000, 100000], ['0', '1', '100', '10⁴', '10⁵'])
    ax.set_xlabel('One-to-one errors / million called founder alleles\n(symlog; linear from 0 to 1)')
    panel(ax, 'a  Final founder_refit components')
    ax = axes[1]
    # Coincident points are deliberately visible as repeated observations, not density estimates.
    for r in records:
        color, marker, size = special(r)
        ax.scatter(int(r['truth_founders']), int(r['founder_rows']), c=color, marker=marker,
                   s=size + 8, alpha=.65, edgecolors='white', linewidths=.25)
    high = max(max(int(r['truth_founders']), int(r['founder_rows'])) for r in records)
    ax.plot([0, high + 1], [0, high + 1], '--', color=GREY, linewidth=.7)
    ax.set(xlim=(0, high + 2), ylim=(0, high + 2), xlabel='Truth founder haplotypes (count)',
           ylabel='Reconstructed founder rows (count)')
    ax.set_aspect('equal')
    ax.legend(handles=[Line2D([], [], color=BLUE, marker='o', ls='', label='Other designs'),
                       Line2D([], [], color=PURPLE, marker='D', ls='', label='Stress n80'),
                       Line2D([], [], color=ORANGE, marker='*', ls='', label='K = 30')],
              frameon=False, loc='upper left', fontsize=6.8)
    panel(ax, 'b  Final component founder counts')
    save(fig, out, 'fig4_founder_endpoint')


def figure_robustness(data, out):
    records = table(data, 'robustness.csv')
    families = ['founder_counts', 'release_segments', 'founder_alleles', 'paintings', 'relatedness', 'pedigree',
                'phase_summary', 'sample_GT_PS', 'recombination']
    labels = ['Founder\ncounts', 'Release\nsegments', 'Founder\nalleles', 'Founder\npaintings', 'Relatedness\ntables',
              'Pedigree\ntables', 'Phase\nsummary', 'Sample\nGT + PS', 'Recombination\ntables']
    variants = VARIANTS[1:]
    designs = ['tromau', 'actm3', 'std_n80', 'stress_n80']
    rowkeys = [(d, v) for d in designs for v in variants]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 8.2), sharey=True, layout='constrained')
    cmap = colors.ListedColormap([ORANGE, '#DDECF5', '#F0F0F0'])
    for ax, version in zip(axes, ['4.1', '4.2']):
        index = {(r['design'], r['variant'], r['output_family']): int(r['identical'])
                 for r in records if r['version'] == version}
        values = np.array([[index.get((d, v, f), 2) for f in families] for d, v in rowkeys])
        ax.imshow(values, cmap=cmap, vmin=0, vmax=2, aspect='auto', interpolation='nearest')
        for i in range(len(rowkeys)):
            for j in range(len(families)):
                val = values[i, j]
                ax.text(j, i, '≠' if val == 0 else '=' if val == 1 else '—',
                        ha='center', va='center', color='white' if val == 0 else INK, fontsize=9)
        for boundary in range(6, len(rowkeys), 6):
            ax.axhline(boundary - .5, color='white', linewidth=2)
        ax.set_xticks(range(len(labels)), labels, rotation=55, ha='right', fontsize=6.8)
        ax.set_yticks(range(len(rowkeys)), [f'{design_name(d).replace(" ◆", "")} · {VNAME[v]}' for d, v in rowkeys], fontsize=6.9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel(ax, f'{"a" if version == "4.1" else "b"}  Release {version}: variant versus reference')
    fig.legend(handles=[Patch(facecolor='#DDECF5', label='=  Identical under stated comparison'),
                        Patch(facecolor=ORANGE, label='≠  Difference'), Patch(facecolor='#F0F0F0', label='—  Not run')],
               loc='outside lower center', ncol=3, frameon=False, fontsize=7)
    save(fig, out, 'fig5_perturbation_agreement')


def figure_mosaic(data, out):
    people = sorted(table(data, 'tromau_samples.csv'), key=lambda r: int(r['plot_order']))
    records = table(data, 'tromau_mosaic.csv')
    chrom = records[0]['chromosome']
    by_sample = defaultdict(list)
    for r in records:
        by_sample[r['sample']].append(r)
    high = max(int(r['end_bp']) for r in records) / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 10.0), sharex=True, layout='constrained')
    for a, ax in enumerate(axes):
        group = people[a * 58:(a + 1) * 58]
        for y, person in enumerate(group):
            if y % 2 == 0:
                ax.axhspan(y - .45, y + .45, color='#F3F3F3', zorder=0)
            if y and person['family'] != group[y - 1]['family']:
                ax.axhline(y - .5, color=LIGHT, linewidth=.4)
            for r in by_sample[person['sample']]:
                track = int(r['track'])
                left, right = int(r['start_bp']) / 1e6, int(r['end_bp']) / 1e6
                named = r['status'] == 'SINGLETON_NAMED' and r['label'] != ''
                color = FOUNDERS[int(r['label']) % len(FOUNDERS)] if named else '#BDBDBD'
                # Thousands of clipped SVG hatch fills make Chromium/Skia emit a
                # separate raster-tile pattern for each interval. Rasterize only
                # these dense mosaic patches as one 300-dpi image per axes;
                # labels, ticks, family separators and legend remain vector.
                rect = Rectangle((left, y - .40 + track * .40), right - left, .34,
                                 facecolor=color, edgecolor='none', linewidth=0,
                                 hatch=None if named else '////', rasterized=True)
                ax.add_patch(rect)
        ax.set_ylim(len(group) - .45, -.6)
        ax.set_xlim(0, high)
        ax.set_yticks(range(len(group)), [p['sample'] for p in group], fontsize=7)
        ax.set_xlabel(f'{chrom} position (Mb)')
        ax.tick_params(axis='y', length=0)
        panel(ax, f'{"a" if a == 0 else "b"}  TroMau {chrom} · samples {a * 58 + 1}–{a * 58 + len(group)}')
    handles = [Patch(facecolor=c, label=f'H{i}') for i, c in enumerate(FOUNDERS)]
    handles.append(Patch(facecolor='#BDBDBD', hatch='////', label='Ambiguous / unnamed'))
    fig.legend(handles=handles, loc='outside lower center', ncol=5, frameon=False)
    save(fig, out, 'fig6_tromau_mosaics', raster_dpi=300)


def figure_recombination(data, out):
    records = table(data, 'tromau_recombination.csv')
    meioses = table(data, 'tromau_meioses.csv')
    chromosomes = list(dict.fromkeys(r['chromosome'] for r in records))
    fig = plt.figure(figsize=(7.5, 10.1), layout='constrained')
    gs = fig.add_gridspec(7, 4, height_ratios=[1] * 6 + [1.7])
    rate_max = max(number(r['rate_cM_per_Mb']) for r in records if r['rate_cM_per_Mb'])
    for i, chrom in enumerate(chromosomes):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        group = [r for r in records if r['chromosome'] == chrom]
        for r in group:
            rate = number(r['rate_cM_per_Mb'])
            left, right = number(r['left_bp']) / 1e6, number(r['right_bp']) / 1e6
            if math.isfinite(rate):
                ax.add_patch(Rectangle((left, 0), right - left, rate, facecolor=BLUE, edgecolor='none'))
            else:
                ax.axvspan(left, right, facecolor=LIGHT, hatch='///', edgecolor=GREY, linewidth=0)
        ax.set_ylim(0, rate_max * 1.05)
        ax.set_xlim(0, max(number(r['right_bp']) for r in group) / 1e6)
        ax.set_title(('a  ' if i == 0 else '') + chrom, fontsize=8, pad=2)
        ax.tick_params(labelsize=6.8, length=2)
        if i % 4 == 0:
            ax.set_ylabel('cM / Mb', fontsize=7)
        if i >= 18:
            ax.set_xlabel('Position (Mb)', fontsize=7)
    for i in [22, 23]:
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.axis('off')
    totals = defaultdict(lambda: [0, 0.0])
    for r in meioses:
        key = (r['parent'], r['child'], r['parent_slot'])
        totals[key][0] += int(r['called_crossovers'])
        totals[key][1] += number(r['exposure_meiosis_bp'])
    values = [v[0] for v in totals.values()]
    ax = fig.add_subplot(gs[6, :2])
    ax.hist(values, bins=np.arange(max(values) + 2) - .5, color=BLUE, edgecolor='white', linewidth=.4)
    ax.set_xlabel('Called crossovers per parent–child meiosis\n(sum over all 22 chromosomes)')
    ax.set_ylabel('Meioses (count)')
    ax.set_ylim(bottom=0)
    panel(ax, 'b  Genome-wide crossover burden')
    ax = fig.add_subplot(gs[6, 2:])
    grouped = []
    for chrom in chromosomes:
        group = [r for r in meioses if r['chromosome'] == chrom]
        grouped.append(sum(int(r['called_crossovers']) for r in group) / len(group))
    ax.bar(range(len(chromosomes)), grouped, color=BLUE, width=.8)
    ax.set_xticks(range(len(chromosomes)), [c.removeprefix('chr') for c in chromosomes], rotation=90, fontsize=6)
    ax.set_xlabel('Chromosome')
    ax.set_ylabel('Called crossovers / reported meiosis')
    ax.set_ylim(bottom=0)
    panel(ax, 'c  Chromosome-specific means')
    save(fig, out, 'fig7_tromau_recombination')


def figure_founder_counts(data, out):
    records = table(data, 'tromau_founder_counts.csv')
    ref = [r for r in records if r['version'] == '4.2' and r['case'] == 'tromau']
    chromosomes = list(dict.fromkeys(r['chromosome'] for r in ref))
    fig = plt.figure(figsize=(7.5, 6.7), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[4.5, 1.45], width_ratios=[1, 1.6])
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    high = max(int(r['end_bp']) for r in ref) / 1e6
    for y, chrom in enumerate(chromosomes):
        group = [r for r in ref if r['chromosome'] == chrom]
        counts = [int(r['founder_rows']) for r in group]
        if y % 2 == 0:
            for ax in axes:
                ax.axhspan(y - .5, y + .5, color='#F3F3F3', zorder=0)
        axes[0].plot([min(counts), max(counts)], [y, y], color=BLUE, lw=1)
        for j, r in enumerate(group):
            axes[0].scatter(int(r['founder_rows']), y + (j - (len(group) - 1) / 2) * .14,
                            color=BLUE, s=20, zorder=3)
            left, right = int(r['start_bp']) / 1e6, int(r['end_bp']) / 1e6
            axes[1].add_patch(Rectangle((left, y - .29), right - left, .58, facecolor='#C2DCEB',
                                       edgecolor=BLUE, linewidth=.6,
                                       hatch='///' if r['ambiguous'] == 'True' else None))
            if right - left > 1:
                axes[1].text((left + right) / 2, y, r['founder_rows'], ha='center', va='center', fontsize=7)
    axes[0].set_xlim(0, max(int(r['founder_rows']) for r in ref) + 2)
    axes[0].set_xlabel('Founder rows per release segment')
    axes[1].set_xlim(0, high * 1.03)
    axes[1].set_xlabel('Release segment (Mb); labels = founder rows')
    for ax in axes:
        ax.set_ylim(len(chromosomes) - .4, -.6)
        ax.set_yticks(range(len(chromosomes)), chromosomes)
        ax.tick_params(axis='y', length=0)
    panel(axes[0], 'a  Inferred founder count')
    panel(axes[1], 'b  Exported release segments')
    ax = fig.add_subplot(gs[1, :])
    labels = []
    for i, (version, case) in enumerate([('4.1', 'tromau'), ('4.1', 'zero_tromau_rev'),
                                       ('4.2', 'tromau'), ('4.2', 'zero_tromau_rev')]):
        group = [r for r in records if r['version'] == version and r['case'] == case and r['chromosome'] == 'chr3']
        convention = 'reference' if case == 'tromau' else 'reversed'
        labels.append(f'{version} {convention} ({len(group)})')
        for j, r in enumerate(group):
            left, right = int(r['start_bp']) / 1e6, int(r['end_bp']) / 1e6
            ax.add_patch(Rectangle((left, i - .28), right - left, .56,
                                  facecolor=BLUE if j % 2 == 0 else SKY, edgecolor='none'))
            if j:
                ax.plot([left, left], [i - .32, i + .32], color=ORANGE, linewidth=1.1)
    ax.set_yticks(range(4), labels)
    ax.set_ylim(3.6, -.6)
    ax.set_xlim(0, max(int(r['end_bp']) for r in records if r['chromosome'] == 'chr3') / 1e6 * 1.02)
    ax.set_xlabel('chr3 position (Mb); orange marks = release boundaries')
    ax.tick_params(axis='y', length=0)
    panel(ax, 'c  chr3 release structure depends on the convention')
    save(fig, out, 'fig8_tromau_founder_counts')


def figure_pedigree(data, out):
    people = sorted(table(data, 'tromau_samples.csv'), key=lambda r: int(r['plot_order']))
    children = [r for r in people if r['parent1'] or r['parent2']]
    parents = sorted({r[key] for r in children for key in ['parent1', 'parent2'] if r[key]})
    families = Counter(r['family'] for r in people)
    states = Counter(r['parent_state'] for r in people)
    fig = plt.figure(figsize=(7.5, 10.0), layout='constrained')
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 4.6], width_ratios=[2.8, 1, .9])
    ax = fig.add_subplot(gs[0, :2])
    hist = Counter(families.values())
    ax.bar(list(hist), list(hist.values()), color=BLUE)
    ax.set(xlim=(0, max(hist) + 1), ylim=(0, max(hist.values()) * 1.17),
           xlabel='Members per inferred relatedness family', ylabel='Families (count)')
    ax.text(.98, .95, f'{len(people)} samples; {len(families)} families', transform=ax.transAxes,
            ha='right', va='top', fontsize=7)
    panel(ax, 'a  Inferred family sizes')
    ax = fig.add_subplot(gs[0, 2])
    state_order = ['zero_observed_parents', 'one_observed_parent', 'two_observed_parents', 'unresolved']
    labels = ['0 observed', '1 observed', '2 observed', 'Unresolved']
    ax.barh(range(4), [states[s] for s in state_order], color=[GREY, SKY, BLUE, ORANGE])
    for i, s in enumerate(state_order):
        ax.text(states[s] + 1, i, str(states[s]), va='center', fontsize=7)
    ax.set_yticks(range(4), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, max(states.values()) * 1.3)
    ax.set_xlabel('Samples (count)')
    panel(ax, 'b  Tier B states')
    ax = fig.add_subplot(gs[1, :2])
    matrix = np.zeros((len(children), len(parents)))
    for i, child in enumerate(children):
        for key in ['parent1', 'parent2']:
            if child[key]:
                matrix[i, parents.index(child[key])] = 1
    ax.imshow(matrix, cmap=colors.ListedColormap(['#FAFAFA', BLUE]), vmin=0, vmax=1,
              aspect='auto', interpolation='nearest')
    for i in range(1, len(children)):
        if children[i]['family'] != children[i - 1]['family']:
            ax.axhline(i - .5, color=LIGHT, linewidth=.6)
    ax.set_yticks(range(len(children)), [r['sample'] for r in children], fontsize=7)
    ax.set_xticks(range(len(parents)), parents, rotation=90, fontsize=7)
    ax.tick_params(length=0)
    ax.set_xlabel('Inferred observed parent (sample identifier)')
    ax.set_ylabel('Inferred child (sample identifier); grouped by relatedness family')
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel(ax, 'c  Tier B inferred parent–child edges')
    ax = fig.add_subplot(gs[1, 2])
    for i, child in enumerate(children):
        ax.scatter(number(child['configuration_bootstrap']), i, color=BLUE, marker='o', s=12)
        ax.scatter(number(child['configuration_loco']), i, color=ORANGE, marker='|', s=25)
    ax.set_ylim(len(children) - .5, -.5)
    ax.set_xlim(0, 1.04)
    ax.set_yticks([])
    ax.set_xticks([0, .5, 1])
    ax.set_xlabel('Selected graph\nconfiguration support')
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=BLUE, label='Bootstrap'),
                       Line2D([], [], marker='|', ls='', color=ORANGE, label='Leave-one-\nchromosome-out')],
              loc='lower left', frameon=False, fontsize=6)
    panel(ax, 'd  Support')
    save(fig, out, 'fig9_tromau_pedigree')


def figure_times(data, out):
    records = table(data, 'whole_run_times.csv')
    cases = table(data, 'cases.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 9.4), sharey=True, layout='constrained')
    high = max(number(r['wall_seconds']) / 60 for r in records)
    for ax, processor, cores in zip(axes, ['Sapphire Rapids', 'Ice Lake'], [112, 76]):
        rows_background(ax, cases)
        for i, case in enumerate(cases):
            matches = [r for r in records if r['case'] == case['case'] and r['processor'] == processor]
            if len(matches) == 2:
                values = [number(r['wall_seconds']) / 60 for r in matches]
                ax.plot(values, [i, i], color='#BDBDBD', lw=.8)
            for r in matches:
                is42 = r['version'] == '4.2'
                ax.scatter(number(r['wall_seconds']) / 60, i, color=BLUE if is42 else ORANGE,
                           marker='s' if is42 else 'o', s=17, zorder=3)
        ax.set_xlim(0, high * 1.05)
        ax.set_xlabel('Whole-run wall time (min)\n(init and evaluation excluded)')
        panel(ax, f'{"a" if cores == 112 else "b"}  {processor} · {cores} cores')
    axes[0].set_yticks(range(len(cases)), [case_name(r) for r in cases], fontsize=6.6)
    fig.legend(handles=[Line2D([], [], marker='o', color=ORANGE, ls='', label='Release 4.1'),
                        Line2D([], [], marker='s', color=BLUE, ls='', label='Release 4.2')],
               loc='outside lower center', ncol=2, frameon=False)
    save(fig, out, 'fig10_whole_run_cost')


def figure_paired(data, out):
    records = table(data, 'paired_costs.csv')
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.3), layout='constrained')
    panels = [
        ('K30_whole_run', 'wall_seconds', 1 / 60, 'a  K = 30 · whole run', 'Wall time (min)'),
        ('N2503_pedigree', 'decision_seconds', 1, 'b  N = 2,503 · decision', 'Decision time (s)'),
        ('TroMau_processor', 'wall_seconds', 1 / 60, 'c  TroMau 4.2 · whole run', 'Wall time (min)'),
        ('K30_whole_run', 'peak_rss_kib', 1024 / 1e9, 'd  K = 30 · whole run', 'Peak RSS (GB)'),
        ('N2503_pedigree', 'peak_rss_kib', 1024 / 1e9, 'e  N = 2,503 · replay', 'Peak RSS (GB)'),
        ('TroMau_processor', 'peak_rss_kib', 1024 / 1e9, 'f  TroMau 4.2 · whole run', 'Peak RSS (GB)'),
    ]
    for ax, (experiment, measurement, scale, title, ylabel) in zip(axes.flat, panels):
        group = [r for r in records if r['experiment'] == experiment and r['measurement'] == measurement]
        if experiment == 'TroMau_processor':
            group.sort(key=lambda r: ['Ice Lake', 'Sapphire Rapids'].index(r['processor']))
            labels = [f'{r["processor"]}\n{r["cores"]} cores' for r in group]
            palette, xlabel = [BLUE, SKY], 'Processor · release 4.2'
        else:
            group.sort(key=lambda r: r['version'])
            labels = [r['version'] for r in group]
            palette, xlabel = [ORANGE, BLUE], 'Release'
        values = [number(r['value']) * scale for r in group]
        bars = ax.bar([0, 1], values, width=.55, color=palette)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.025,
                    f'{value:.2f}' if 'RSS' in ylabel else f'{value:.1f}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks([0, 1], labels)
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        panel(ax, title)
    save(fig, out, 'fig11_matched_cost')


def figure_k30(data, out):
    records = [r for r in table(data, 'founder_components.csv') if r['case'] == 'k30']
    accuracy = [r for r in table(data, 'simulation_accuracy.csv') if r['case'] == 'k30']
    contigs = list(dict.fromkeys(r['contig'] for r in records))
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.9), layout='constrained')
    for ax, key, title, ylabel in [
        (axes[0, 0], 'called_allele_errors', 'a  Component-matched founder errors', 'Called founder allele errors (count)'),
        (axes[0, 1], 'chimeric_segment_charges', 'b  Long-range founder mismatch', 'Chimeric segment charges (count)'),
        (axes[1, 0], 'founder_rows', 'c  Founder recovery', 'Founder rows per component (count)')]:
        for version, offset, color in [('4.1', -.18, ORANGE), ('4.2', .18, BLUE)]:
            values = [sum(int(r[key]) for r in records if r['version'] == version and r['contig'] == c) for c in contigs]
            ax.bar(np.arange(len(contigs)) + offset, values, .34, color=color, label=version)
        if key == 'founder_rows':
            truth = [int(next(r for r in records if r['contig'] == c)['truth_founders']) for c in contigs]
            ax.scatter(range(len(contigs)), truth, marker='_', color=INK, s=250, zorder=4, label='Truth')
        ax.set_xticks(range(len(contigs)), contigs)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Simulation contig')
        panel(ax, title)
    axes[0, 0].legend(title='Release', frameon=False, fontsize=7)
    axes[1, 0].legend(frameon=False, fontsize=7, loc='lower left')
    ax = axes[1, 1]
    a = {r['version']: r for r in accuracy}
    fields = [('Genotype errors', 'genotype_errors', '{:.0f}'),
              ('Phased share (%)', 'phased_share', '{:.2f}'),
              ('Switch errors', 'switch_errors', '{:.0f}'),
              ('Released phase sets', 'released_phase_sets', '{:.0f}'),
              ('Correct pedigree edges', 'correct_edges', '{:.0f}'),
              ('Missed pedigree edges', 'missed_edges', '{:.0f}')]
    ax.axis('off')
    ax.text(.02, .92, 'Metric', weight='bold', fontsize=8)
    ax.text(.71, .92, '4.1', color=ORANGE, weight='bold', ha='right', fontsize=8)
    ax.text(.99, .92, '4.2', color=BLUE, weight='bold', ha='right', fontsize=8)
    for i, (label, key, fmt) in enumerate(fields):
        y = .77 - .127 * i
        ax.text(.02, y, label, fontsize=7.4)
        for version, x in [('4.1', .71), ('4.2', .99)]:
            value = number(a[version][key]) * (100 if key == 'phased_share' else 1)
            ax.text(x, y, fmt.format(value), ha='right', fontsize=7.4)
    panel(ax, 'd  Downstream changes are mixed')
    save(fig, out, 'fig12_k30_recovery')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PAPER / 'data')
    parser.add_argument('--output-dir', type=Path, default=PAPER / 'figures')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.data_dir / 'provenance.json').read_text())
    if metadata['schema'] != 1:
        raise ValueError('Unsupported summary table schema')
    for plot in [figure_accuracy, figure_founder_endpoint, figure_robustness, figure_mosaic,
                 figure_recombination, figure_founder_counts, figure_pedigree, figure_times,
                 figure_paired, figure_k30]:
        plot(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
