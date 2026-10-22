# Simulation and evaluation

A simulation generates a cross with a known pedigree from founder templates, sequences its observed individuals
at low coverage, and runs the same pipeline on the simulated allele depths. Inference receives neither the
generating pedigree, nor cohort labels, nor the generating read model or map; truth is read only by
`haplotypes evaluate`, after inference, and never fed back. Founder accuracy, sample-phase accuracy, pedigree
accuracy and map accuracy are different quantities and are reported separately.

## A first simulation

`haplotypes example` writes a self-contained example: six founder haplotypes of independent Bernoulli(1/2) SNPs on
three short contigs (`DEST/templates/`), a complete configuration (`DEST/simulation.toml`: cohorts of 20, 30 and
30 individuals at depth 5) and a README.

```bash
haplotypes example work/examples/quick --seed 6100
haplotypes run work/examples/quick/simulation.toml
haplotypes evaluate work/examples/quick/run --stages all
```

The example is a software check, not an empirical diversity model or an accuracy benchmark: three short contigs
can leave the whole pedigree unresolved (the genome-wide stages need at least three contigs, and Tier-B calls at
least five informative chromosomes). `haplotypes example DEST --contigs 22 --sites 2400 --length-bp 4800000`
writes a larger synthetic genome. For scientific comparisons use fixed, biologically appropriate templates and
enough chromosomes.

## Templates

`[simulation] templates` is a directory with one `<contig>.npz` per simulated contig:

- `positions`: strictly increasing 1-based marker coordinates, shape `(sites,)`;
- `allele_probabilities`: founder-sequence probabilities `(P(REF), P(ALT))`, shape `(haplotypes, sites, 2)`;
  consecutive pairs of haplotypes form the founders, so the number of haplotypes is even.

Exact probability ties are resolved by seeded draws when the founders are materialised. Templates are inputs that
generate founders, not haplotypes the pipeline reconstructs; template density and contig length define the
scientific problem, so record both. `haplotypes templates --vcf FILE --output DIR [--contigs C ...]` builds
templates naively from a real cross: fixed-error genotype likelihoods without quality control or read calibration,
the discovery block search in 100 kb windows every 50 kb, and a simple linker that chains the window haplotypes
into up to six chromosome-length haplotypes. Reconstruction never uses this linker.

A simulation configuration names the templates and the contigs:

<!-- check_docs: config -->
```toml
[run]
output = "work/runs/sim_seed401"

[input]
kind = "simulation"
contigs = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6"]

[simulation]
templates = "work/data/simulation_templates"
seed = 401
```

## Generating designs

`[simulation]` and `[simulation.reads]` change data generation only; [configuration.md](configuration.md) lists
their defaults. The pedigree starts from the founders: cohort 1 descends from them, and every later cohort from
pairs of members of the previous cohort.

| setting | meaning |
| --- | --- |
| `cohorts` | individuals in successive cohorts (default 20, 100, 200) |
| `backcross_fraction`, `backcross_start_cohort` | probability that a child of a cohort at or after the start cohort is a backcross: a member of the previous cohort mated with one of its own actual parents |
| `observe_cohorts` | sequence only these cohorts (numbered from 1; empty: all) |
| `observed_fraction` | random fraction of those individuals sequenced, drawn after the whole pedigree exists |
| `depth` | mean read depth per site |
| `generating_rate_cm_per_mb`, `generating_map` | the generating genetic map, independent of the inference map |
| `seed` | an integer, or `"random"` (drawn once; the realised seed is recorded) |

The complete pedigree is generated before the observed individuals are selected. Unobserved individuals are
removed from the reads and every inference input; they remain in the truth. M0, M1 and M2 therefore count observed
parents: the 20 individuals of cohort 1 have unsequenced parents and are true M0. The run file's
`simulation/manifest` record holds the realised seed, the observed truth pedigree and the generated pedigree;
`OUTPUT/simulation/truth_pedigree.csv` lists the analysed samples with their cohort and true parents.

Two hundred generated individuals, exactly 160 observed:

```toml
[simulation]
seed = 6200
cohorts = [20, 80, 100]
observed_fraction = 0.8
```

Only the last three of six cohorts observed:

```toml
[simulation]
seed = 6201
cohorts = [20, 30, 30, 100, 100, 100]
observe_cohorts = [4, 5, 6]
```

Half of the third cohort's matings are backcrosses to an actual parent:

```toml
[simulation]
seed = 6202
cohorts = [20, 30, 30]
backcross_fraction = 0.5
backcross_start_cohort = 3
```

### Reads

Per cell, depth ~ Poisson(depth x sample factor) and the alternate reads ~ Binomial(depth, p) with p =
`error_rate`, `heterozygote_alt_probability` or 1 - `error_rate` for genotypes 0, 1, 2. `depth_cv` draws a
lognormal sample factor (mean one) per sample and contig. `dropout_fraction` removes one contiguous tract of that
fraction of the markers per sample and contig, and raises the depth elsewhere so that the expected depth over all
markers stays at `depth`; dropout is counted in markers, not base pairs. With `read_length_bp > 0` the reads come
from paired-end fragments (insert length Normal(`fragment_mean_bp`, `fragment_sd_bp`)), so sites closer than a
fragment share reads while each site keeps the Poisson depth marginal.

A combined stress control at 5x might use:

```toml
[simulation.reads]
depth_cv = 0.6
dropout_fraction = 0.1
heterozygote_alt_probability = 0.6
error_rate = 0.04
```

Inference fits its read model from the observed allele depths ([evidence](methods/evidence.md)); the generating
parameters are never handed to it (`--set quality.read_calibration=false` selects the fixed-error model). Use
separate perturbations to attribute a failure: a combined control measures only the joint effect. These controls
do not replace real sequencing data, mapping-bias models or breeding records.

### Random streams

A configuration and seed always give the same data, at any number of workers: the founders of each contig, the
pedigree structure, the gametes of each child on each contig (inherited homologue, crossovers, mutations), the
reads and the observation choice draw from separate seeded streams fixed before any scheduling. The contig order
is part of the streams. Without a generating map, gaps between crossovers are exponential at the generating rate;
with one, a Poisson number of crossovers is placed uniformly in genetic distance. Mutations flip alleles at
`SimulationConfig.mutation_rate_per_bp = 1e-10` per meiosis.

## Evaluation

```bash
haplotypes evaluate RUN --stages all --cores 32
haplotypes evaluate RUN --contigs chr3 chr10 --stages discovery local_selection_l1 local_selection_l2
```

`--stages available` (default) evaluates the latest founder product of each contig and every downstream product
present; `all` every founder product present; explicit stages require their products. The founder stages, in
pipeline order, are `discovery` (raw blocks), `local_selection_initial`, `local_selection_l1`,
`local_selection_l2`, `assembly_l1` ... `assembly_l4` (the final assembly's joined levels),
`assembly` (the final joined product; no founder-path refinement runs), `painting` (the panels after label continuity) and `founder_refit`; the downstream stages
are `pedigree`, `family_phase` and `recombination`. A hierarchy level that did not run is not evaluated.
`--cores` is the evaluation's CPU budget; chromosomes run in parallel within memory estimates.

Evaluation reads each contig's `simulation/truth` record (founder haplotypes, painting, alleles, crossovers) and
the products it compares, never the reads or likelihoods. It writes `RUN/evaluation/`:

| file | contents |
| --- | --- |
| `summary.json` | seed, samples, pedigree counts, per-chromosome and per-stage rows, totals, interpretation notes |
| `chromosomes.csv` | per chromosome: family-phase genotype and phase metrics, map metrics |
| `founders.csv` | per chromosome and founder stage: totals and denominators |
| `founder_components.csv` | per component (or block) of every founder stage |
| `founder_matches.csv` | the one-to-one row-to-founder matches behind `founder_components.csv` |

### Founder metrics

- `called_allele_errors`: each reconstructed row is matched to its nearest truth founder over its whole component;
  unknown alleles are excluded from the called denominator, so duplicate rows can still have good precision.
- `one_to_one_called_allele_errors`, `truth_to_panel_errors_missing`: a one-to-one matching minimises mismatches
  plus unknowns and charges an absent founder its whole component length, so incomplete panels show; extra rows and
  their alleles are reported separately (`unmatched_founder_rows`, `unmatched_called_alleles`) and never raise
  completeness. Founder labels are matched per component, never re-matched at single sites.
- `segment_allele_errors` relabels each founder row segment-wise (a label switch costs 20 mismatches);
  `chimeric_segment_charges` = `called_allele_errors` - `segment_allele_errors` (charged because a row follows
  another founder for a segment), and `segment_label_switches` counts those segments.
- Represented and absent strata use the one-to-one matching: represented ancestry is carried by at least one
  observed individual, which does not make it independently identifiable.

### Phase, pedigree and map metrics

- `genotype_errors` over `called_genotypes`; `phase_switch_errors` between eligible adjacent true heterozygotes of a
  sample; unsupported components and missing or wrong intervening heterozygotes break the comparison.
  `component_aligned_allele_errors` allow one arbitrary strand swap per component.
- `long_phase_switches_min5` and `long_phase_switches_min20` count orientation changes between runs of
  correct-genotype true heterozygotes (uncalled or wrong ones bridged) when both runs hold at least 5 or 20
  heterozygotes; single-site flips, which dominate `phase_switch_errors`, are excluded.
- Pedigree: exact observed-parent configurations and edges of the Tier-B calls (correct, extra, missing), per true
  parent-count state.
- Maps: true crossovers are counted only on correctly inferred edges within the inferred observable exposure (not
  whole-genome sensitivity). `called_interval_hit_rate` is the fraction of called crossover intervals on correct
  edges that contain a true crossover; `reported_interval_hit_rate` applies the rule to the bin-honest reported
  intervals; read both with `called_interval_median_width_bp` and `reported_interval_median_width_bp`, since a
  wider interval contains more by construction.

## Exports of simulated runs

Simulated alleles are abstract 0/1 values without REF/ALT letters, so the export writes them as A (0) and C (1),
labelled abstract in the VCF headers and the manifest. A simulated run's `OUTPUT/export/` holds its founder and
sample VCFs like any other run's; to write them elsewhere:

```bash
haplotypes export work/examples/quick/run work/exports/quick
```

[usage.md](usage.md#export) describes the export files.

## Recording a run

Keep the run directory's `config.toml` (the resolved configuration, seed included) and `run_records/` (the package
source digest and commit, dependency versions, host and CPU affinity, exclusive stage timings), the templates and
the source commit. A seed alone does not reproduce a run with other templates, code or settings. Add exclusive
stage times to compare stages: local selection runs assemblies inside it, so inclusive times count work twice. A
resumed invocation's times include resume overhead and do not measure a fresh run.
