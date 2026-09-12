# Running and resuming

Use `python run.py COMMAND --help`. Command-line values override TOML values;
omitted settings use documented defaults. Configuration paths are resolved
relative to the working directory, not the TOML file. Examples assume the
repository root. No environment installation is needed when the project
dependencies are already available. An installed package exposes the same CLI
as `haplotypes`.

## Inputs

Real workflows accept indexed VCF/BCF files and the corresponding cross-design
workbook. The workbook supplies sample eligibility/chronology, not individual
parentage truth. Check the input paths in `configs/astcal.toml` or
`configs/tropheops.toml`. Data are intentionally not distributed in Git.

Simulations accept a directory with one `CONTIG.npz` per chromosome:

- `positions`: strictly increasing marker coordinates, shape `(sites,)`.
- `allele_probabilities`: founder-sequence probabilities, shape
  `(haplotypes, sites, 2)`; an even number of haplotypes is paired into parents.

Probability ties are resolved with seeded unbiased draws when concrete
simulation truth is generated. These templates are input sequences, not the
haplotypes subsequently discovered from simulated reads. The existing local
templates contain 8,956,852 markers on 22 cichlid chromosomes (`chr1`–`chr20`,
`chr22`, `chr23`); the absent `chr21` label is intentional.

Alternatively, `simulate --vcf PATH` derives empirical templates from that
VCF/BCF using the template-building route. This can be expensive. Supply
`--templates` to reuse frozen sequence inputs while still regenerating reads,
discovery, assembly and all downstream inference for each new seed.

The default simulation has cohorts of 20, 100 and 200 individuals, depth 5,
read-error probability 0.02 and a generating rate of 5 cM/Mb. Its first cohort's
biological parents are unsequenced, yielding 20 M0 and 300 M2 true observed-parent
states. Inference receives neither the generating pedigree nor cohort labels.
The generating map and inference map are separate settings.

## Simulation stage boundaries and chromosome shards

`simulate --stop-after-stage 01_blocks` stops after block discovery;
`--stop-after-stage 09_painting` stops after L1–L4 assembly and painting.
Both retain completed checkpoints. Omit the option on a later invocation to
resume downstream work.

`--process-contigs chr1 chr2` processes only those chromosomes from an existing
simulation, in the cached manifest order. It does **not** change the simulated
input chromosome set: keep the original `--contigs` / `[inputs].contigs`, seed,
inputs and scientific settings. Globally complete founder-template and
simulated-read checkpoints must already exist; a shard cannot initialize them.

For example, after the shared inputs have completed, run non-overlapping shards
on separately allocated nodes against the same run directory:

```bash
python run.py simulate --config configs/simulation.toml --seed 400 \
  --process-contigs chr1 chr2 --stop-after-stage 09_painting
```

Specify a stop boundary for shard workers, before genome-wide pedigree/family
inference. Shards retain chromosome checkpoints but do not publish global stage
completion. After every shard finishes, run once without shard or stop controls
to validate/reuse the chromosome outputs and continue genome-wide inference.

The TOML equivalents are `[run].process_contigs = ["chr1", "chr2"]` and
`[run].stop_after_stage = "09_painting"`. Existing environment controls
`BHD_SIM_CONTIGS=chr1,chr2` and `BHD_SIM_STOP_AFTER_STAGE=09_painting` also work.
Precedence for these controls is CLI > TOML > environment > no restriction.
Remove the TOML controls and unset the environment variables when resuming a
full run; omitting CLI flags alone does not override those fallbacks.

For shared-family recombination evidence, precedence is likewise CLI >
`[recombination].shared_family_evidence` > `BHD_RECOMBINATION_SHARED_FAMILY` > on.
The environment accepts `1/0`, `true/false`, `yes/no` and `on/off`, ignoring case
and surrounding whitespace. Invalid values are rejected rather than enabling
the feature accidentally. Use `--no-shared-family-evidence` to explicitly
disable it even when the selected TOML example sets it to true.

## Assembly transition models

The default, `--assembly-model dense`, retains arbitrary dense learned
transitions and uses bounded candidate-panel search at every assembly level.
Its dominant founder-count dependence is cubic. For larger founder panels,
`--assembly-model structured` selects sparse-specific plus positive-background
transitions, giving near-quadratic scaling for fixed fitting/search budgets.
Both modes use the same bounded search (16 full scores per proposal category)
and the 20-iteration linker limit.

```bash
python run.py simulate --config configs/simulation.toml --seed 400 \
  --assembly-model structured --output work/runs/structured_seed_400
```

The same option is available for `astcal` and `tropheops`. Set
`[run].assembly_model = "dense"` or `"structured"` in TOML, or use
`HAPLOTYPES_ASSEMBLY_MODEL`. Precedence is CLI > TOML > environment > dense.
There is no automatic founder-count cutoff: the user chooses the model.
Structured transitions are a restricted statistical model, not an exact
acceleration of arbitrary dense transitions. Read the
[scaling and accuracy trade-offs](founder_scaling.md).

Assembly selection does **not** change Stage 1. Its established search remains
the default. The separate `--discovery-search batched` option is experimental;
its TOML/environment settings are `[run].discovery_search` and
`HAPLOTYPES_DISCOVERY_SEARCH`, with `standard` as the default. An earlier
combined discovery/assembly experiment had substantial accuracy regressions;
do not enable batched discovery merely to obtain structured assembly.

Use a separate output directory when changing scientific settings. Existing
checkpoint identities include the actual assembly configuration and reject
incompatible reuse; no manual cache relabeling is needed.

## Checkpoints and outputs

By default, a simulation writes `work/runs/seed_<seed>/`. Real-data defaults are
`work/runs/astcal/` and `work/runs/tropheops/`. Use `--output` to isolate an
alternative attempt and `--checkpoints` to override its checkpoint location.
Do not share a checkpoint directory between different seeds or configurations.

| Checkpoint directory | Contents |
| --- | --- |
| `00_founder_templates/` | Frozen simulation sequence inputs |
| `00_simulated_reads/` | Simulated observations, true pedigree, alleles and raw crossover events |
| `01_blocks/` | Discovered 200-SNP block haplotypes, raw likelihoods/observation masks as applicable |
| `00_genotype_evidence/` | Lossless compact GL/position/observation-mask cache for downstream inference |
| `09_painting_release_work/` | Per-chromosome preprocessing and completed L1–L4 assembly phases |
| `09_painting/` | Typed component-local painting products |
| `10_pedigree_evidence/` | Prepared/scored chromosome evidence |
| `10_pedigree/` | Genome-wide inferred pedigree and support tables |
| `11_phase_correction/` | Final phase products, `CONTIG.iterations.p5.b2` work checkpoints, and compact completion summary |
| `12_recombination/` | Conditional map products and completion summary |

Atomic protocol-5/Blosc checkpoints use `.p5.b2`; completion markers are written
only after required outputs exist. Inputs, model/configuration and relevant
source identities are checked at stage boundaries. Interrupted work resumes
from durable chromosome, assembly-phase or family-iteration checkpoints. Source
changes may invalidate affected stages; do not bypass a mismatch by relabeling
an old checkpoint as current.

Reconstruction writes compact genotype evidence after validating its inputs.
T10 and T11 use this derived cache when it matches the source checkpoint files;
older runs without it continue to read their original raw/block checkpoints.
The cache does not replace the rich simulation truth or discovery products:
keep those source files, including linked targets, for validation and resume.
It uses the same likelihood precision and missing-observation mask, trading an
additional write and disk space for smaller repeated reads and lower memory.

T11 uses one phase-focused path, with no separate full-posterior or imputed
source stage. Iteration checkpoints retain family messages, the preceding
polished phase and the consecutive-stability count; large numerical caches are
rebuilt on restart. A 520-iteration safety limit refuses release if phase has
not stabilized. Stable phase does not assert marginal-posterior convergence.

The phase-focused schema changes T11 and downstream cache identities. Existing
T09/T10/raw checkpoints remain reusable; preserve older results and use a new
output/checkpoint root for the changed downstream stages. Do not overwrite or
relabel old T11/T12 checkpoints.

When an attempt reuses validated inputs, its checkpoint directories may be
symbolic links to an earlier attempt. `attempt.json` records that source.
Keep the linked targets—including input stages under `work/history/`—while
the current run or any future resume needs them. A history directory is not
automatically disposable merely because its assembly attempt was superseded.

Readable output directories include `pedigree/`, `phase_correction/`, and
`recombination_map/`. The latter separates posterior rate curves, called
crossover intervals, coverage intervals, per-meiosis diagnostics and plots.
Tables of phase-stability summaries do not replace the detailed allele arrays
in checkpoints.

## Simulation evaluation

```bash
python run.py evaluate --output work/runs/seed_400 --cores 76
```

For runs using the standard `OUTPUT/checkpoints` location, this writes
`evaluation/summary.json` and `evaluation/chromosomes.csv`. Evaluation runs only
after inference and never feeds truth back upstream. It reports exact observed
parent configurations and edges, called coverage, genotype errors, and phase
switches between eligible adjacent true heterozygotes. Unsupported components,
missing or incorrect intervening heterozygotes break switch comparisons.
Component-aligned allele errors allow one arbitrary strand swap per component.
Map comparisons count true crossovers only on correctly inferred edges within
the inferred observable exposure; they are not whole-genome sensitivity claims.

Cold Numba compilation and pool startup can dominate tiny fixtures. Full runs
use a chromosome-at-a-time data path to bound memory; numerical and process
phases do not each receive an independent full-node budget. Concurrent seeds
belong on separate allocated nodes with separate output roots.
