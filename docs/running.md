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
`--stop-after-stage 09_painting` stops after local feedback selection, final
L1–L4 assembly and painting.
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

## Local feedback selection

All three reconstruction commands use this sequence by default:

1. Keep the original missing-aware 200-SNP discovery results.
2. Assemble them through L1 and refit/project that context back to local blocks.
3. Select from original and L1-feedback candidates using the original raw
   likelihoods and observation masks.
4. Assemble those **selected** panels through L1+L2 and refit back to blocks.
5. Select again, using original, selected-L1 and fresh L2-feedback candidates.
6. Run final L1–L4 and the unchanged downstream algorithms on the selected blocks.

Selection always runs after each round; there is no end-only ordering option.

The intermediate passes are proposals, not additional observations. Their
carrier model excludes the current block's emission before refitting its
alleles. Neither truth nor downstream assembly/painting metrics select panels.
The configured assembly model/search applies to the context passes as well as
final assembly. Supplied recombination maps also determine context transitions;
unmapped chromosomes use the fallback rate, scaled by assembly generations.

`--feedback-selection balanced` is the default. It starts from the
cavity-selected feedback-only refit, filters candidates explainable by a single
join between current backbone haplotypes, requires a positive local BIC gain
for additions, then performs same-K cavity-selected allele/confidence refinement.
The final refinement can change or withdraw earlier calls.

`--feedback-selection strict` instead permits only private-allele rescue,
requires the same positive BIC gain, rechecks support for added rows, and
protects the original clean feedback calls. It recovers less missing variation.
That protection applies to the refitted backbone within each round's rescue,
not to freezing all first-round calls during the next context/refit.

```bash
python run.py simulate --config configs/simulation.toml --seed 400 \
  --feedback-selection strict --output work/runs/strict_seed_400
```

The flag works for `astcal` and `tropheops` too. TOML uses
`[run].feedback_selection = "balanced"` or `"strict"`; the environment variable
is `HAPLOTYPES_FEEDBACK_SELECTION`. Precedence is CLI > TOML > environment >
balanced. This is local block feedback, **not** a T11 → T10 feedback loop.

Both modes retain unknown calls. Exact context-site inference is bounded to
at most ten distinct context haplotypes; larger contexts or contexts that do
not cover whole original blocks retain their input local proposals, with a
recorded skip reason. This cap bounds computation, not the biological founder
count. Local selection itself does not impose that ten-founder cap.

In the 9,287-block ordering comparison (six seed/chromosome cases), balanced
selection after each round gave 135 called-allele errors and 566 truth-to-panel
errors/missing, versus 183 and 652 when selecting only at the end. The 135
errors span 10,265,983 called founder alleles: **13.15 errors per million
called SNP alleles (0.001315%)**, excluding unknown calls. These are
local panel metrics, **not** final chromosome or sample phase errors. The
balanced result is a measured precision/variation trade-off, not a uniform
accuracy improvement or a guarantee against per-block regression.

This ordering changes feedback, T09 and downstream cache identities, including
products from the earlier end-only feedback workflow.
Preserve old runs and use a separate output/checkpoint root; compatible original
input/discovery stages can still be linked into that root. Do not relabel old
T09 products. Both feedback rounds and their selection batches are checkpointed.
Balanced and strict can share the initial raw L1 context/proposals, but have
separate first-round selections and second-round contexts/proposals/selections.
Switching modes does not overwrite raw discovery results.

## Assembly transition models and search breadth

The default, `--assembly-model dense`, retains arbitrary dense learned
transitions and defaults to bounded candidate-panel search at every assembly
level. With bounded search, its dominant founder-count dependence is cubic. For larger founder panels,
`--assembly-model structured` selects sparse-specific plus positive-background
transitions, giving near-quadratic scaling for fixed fitting/search budgets.
Both modes default to bounded search (16 full scores per proposal category)
and retain the 20-iteration linker limit.

```bash
python run.py simulate --config configs/simulation.toml --seed 400 \
  --assembly-model structured --output work/runs/structured_seed_400
```

The same option is available for `astcal` and `tropheops`. Set
`[run].assembly_model = "dense"` or `"structured"` in TOML, or use
`HAPLOTYPES_ASSEMBLY_MODEL`. Precedence is CLI > TOML > environment > dense.
There is no automatic founder-count cutoff: the user chooses the model.

Search breadth is a separate choice: `--assembly-search bounded` (default) or
`--assembly-search broad`. The broader mode retains the optimized diversity
beam and broader full-refit panel/chimera search, including Numba scoring,
cached proposal tensors and dynamic CPU allocation. To select the earlier
broad-search/dense-transition combination explicitly:

```bash
python run.py simulate --config configs/simulation.toml --seed 400 \
  --assembly-model dense --assembly-search broad \
  --output work/runs/broad_seed_400
```

This works for `astcal` and `tropheops` too. TOML uses
`[run].assembly_search = "bounded"` or `"broad"`; the environment variable is
`HAPLOTYPES_ASSEMBLY_SEARCH`. Precedence is CLI > TOML > environment > bounded.
The two search modes are heuristics with the same full Viterbi/BIC acceptance
objective; broader search can be slower and is not uniformly more accurate.
Combining `structured` transitions with `broad` search is allowed, but no longer
gives the near-quadratic whole-assembly bound. Search breadth changes L1–L4,
not Stage 1 or downstream models.

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
| `02_feedback_l1_assembly/`, `02_feedback_l1/` | Initial context-assembly levels and raw local proposals, shared between modes |
| `02_feedback_<mode>_l1/` | First-round 128-block selection batches and selected panels |
| `02_feedback_<mode>_l2_assembly/` | Context assembly through L1+L2 from that mode's selected first-round panels |
| `02_feedback_<mode>_l2/` | Second-round raw proposals, 128-block selection batches and final selected panels; `<mode>` is `balanced` or `strict` |
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

Changes confined to T11 can reuse compatible T09/T10/raw checkpoints in a new
downstream output root. The local-feedback change described above also changes
T09 inputs: reuse only compatible original input/discovery stages for that
upgrade. Do not overwrite or relabel old scientific products.

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
