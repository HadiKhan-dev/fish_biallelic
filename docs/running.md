# Running and resuming

Use `python run.py COMMAND --help`. Command-line values override TOML values;
omitted settings use documented defaults. Configuration paths are resolved
relative to the working directory, not the TOML file. Examples assume the
repository root. No environment installation is needed when the project
dependencies are already available. An installed package exposes the same CLI
as `haplotypes`.

## CPU execution

`--cores` sets the total process-by-thread budget within the allocation.
Discovery and hierarchy workers share that budget dynamically at numerical
phase boundaries; freed cores cannot join an already-running kernel. Evidence
validation/snapshotting, worker-array packing, occupancy masks and hierarchy
eligibility also use the explicit budget, before or after the worker pool.
Large shared-memory initialization uses at most 16 copying threads because
additional threads slowed the measured memory-bandwidth-bound operation.
Other numerical work retains the requested budget.

Checkpoint I/O, process startup, ordered panel/beam updates and inter-stage
dependencies can still leave cores idle. This is not a claim of sustained
full-node utilization or independent chromosome sharding within one run.

## Inputs

The general `reconstruct` command accepts indexed, sorted, biallelic SNP
VCF/BCF files with `FORMAT/AD` (REF and ALT read depths), without a workbook.
GT-only and PL-only files are not currently supported by this entry point.
Missing AD is unobserved, not reference support. Duplicate positions,
multiallelic variants and indels must be normalized/filtered before this route;
it reports unsupported records instead of silently truncating them.

```bash
haplotypes reconstruct --vcf cross.bcf --output work/runs/my_cross \
  --contigs chr1 chr2 chr3 --cores 76
```

Select populated physical contigs explicitly if the header also declares empty
scaffolds. At least three contigs are required for the current genome-wide
pedigree model; smaller reconstructions can stop at `block_discovery` or `painting`
with `--stop-after-stage`. Both discovery and downstream algorithms are the
same canonical implementations used by the existing workflows. All assembly,
feedback, CPU-budget and recombination-map options remain available.

Without constraints, all input samples are candidate pedigree members and no
cohort/sex/generation labels are inferred from names. Optional `--eligibility
constraints.json` (TOML: `[inputs].eligibility`) accepts:

```json
{
  "candidate_parents": {"fish_c": ["fish_a", "fish_b"], "fish_d": []},
  "excluded_samples": ["outside_pedigree"],
  "ineligible_children": [],
  "direction_supported_edges": [["fish_a", "fish_c"]]
}
```

Unlisted children retain every non-self candidate. An empty list explicitly
allows no observed parent. Excluded samples cannot be parents or inferred
children; `ineligible_children` may still be parents. Explicit direction edges
are **[parent, child]** and should express independently supported chronology,
not an assumed true individual parent pair. Candidate restrictions do not by
themselves assert direction or parentage. These exclusions affect pedigree
inference, not founder discovery: remove unrelated/outgroup samples from the
VCF itself if they must not contribute to discovery and painting.

The dataset-specific `astcal` and `tropheops` workflows additionally use their
corresponding cross-design workbook. The workbook supplies sample eligibility/chronology, not individual
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
VCF/BCF using the simple empirical template-building linker in
`simulation/templates.py`. This is the retained naive-linking code: it produces
simulation input sequences, **not** the reconstructed offspring haplotypes.
Reconstruction always uses the current shared L1–L4 linker. This template
construction can be expensive. Supply
`--templates` to reuse frozen sequence inputs while still regenerating reads,
discovery, assembly and all downstream inference for each new seed.

The default simulation has cohorts of 20, 100 and 200 individuals, depth 5,
read-error probability 0.02 and a generating rate of 5 cM/Mb. Its first cohort's
biological parents are unsequenced, yielding 20 M0 and 300 M2 true observed-parent
states. Inference receives neither the generating pedigree nor cohort labels.
The generating map and inference map are separate settings.

## Shared inference options

The same options are wired through `simulate`, `reconstruct`, `astcal` and
`tropheops`. No workflow-specific source edit is needed to select them.

| CLI option | Default | What it controls |
| --- | --- | --- |
| `--read-calibration` / `--no-read-calibration` | on | Observed-AD calibration reused from discovery through final phase |
| `--pedigree-calibration predictive\|off` | predictive | Cross-chromosome pedigree evidence scale |
| `--assembly-model dense\|structured` | dense | Block-linkage transition model |
| `--assembly-search bounded\|broad` | bounded | Panel-search breadth, independently of transition model |
| `--feedback-selection path\|balanced\|strict` | path | Initial normalized-path selection and selection after both feedback rounds; balanced/strict are explicit alternatives |
| `--feedback-segment-exchange` / `--no-feedback-segment-exchange` | off | Optional final same-count segment exchange; requires path selection |
| `--founder-refinement on\|off` | on | Progressive final L1–L4 refinement and final count-up/refit |
| `--discovery-search standard\|batched` | standard | Local search; batched remains experimental |
| `--shared-family-evidence` / `--no-shared-family-evidence` | on | Shared phase-error evidence in downstream recombination maps |

For these switches, explicit CLI values override TOML, then the supported
environment settings, then defaults. These TOML keys belong in `[run]`, except
`shared_family_evidence`, which belongs in `[recombination]`. Boolean settings
accept `true/false`, `on/off`, `yes/no` and `1/0` strings, or TOML booleans.
Recombination also exposes its shared-family switch in the standalone command.
Detailed sections below describe the scientific trade-offs and cache effects.

## Read-model calibration

Calibration is on by default for all reconstruction workflows. It fits read
error, heterozygote balance and supported overdispersion from observed AD,
including pooled sample effects, without truth or
pedigree input, and supplies the same raw likelihoods to every inference stage.
Use `--no-read-calibration` (TOML: `[run].read_calibration = false`) for a
fixed-model comparison. Fits, held-out gains and fallback reasons are cached.
See [Read-model calibration](read_calibration.md) for the model, limitations
and checkpoint compatibility.

Pedigree evidence calibration is also **on by default**, using pooled predictive
weighting across chromosomes. Disable it with `--pedigree-calibration off`
(TOML: `[run].pedigree_calibration = "off"`) for unscaled evidence.
Direct workflows also accept
`HAPLOTYPES_PEDIGREE_CALIBRATION=off|predictive`; explicit CLI options override
TOML, which overrides the environment. This learns one chromosome-evidence
weight, not parent-count priors, release thresholds or pedigree directions.
Fewer than eight usable chromosomes, a boundary optimum, optimizer failure or
no predictive gain retains scale 1. The fit and partition names are exported to
`pedigree/evidence_calibration.json` beneath the workflow's output directory
and stored in the pedigree checkpoint. Raw genetic scores remain reusable;
decision checkpoints distinguish the option and fitted result, preserving prior
pedigree decisions. Discovery, assembly, painting and genetic scores need not be
rerun. If the accepted pedigree relationships change, downstream family-phase
and map checkpoints must be recomputed for those changed inputs. This promotion
accepts the aggregate improvement and known individual/cohort regressions in
[Validation](validation.md). See [Methods](methods.md) for the conditional
interpretation and resampling limits.

## Stage vocabulary and checkpoint compatibility

All workflows use descriptive stage names. A fresh checkpoint root contains
`block_discovery/`, the `feedback_*/` contexts/selections,
`assembly/` for internal assembly/refinement work,
`painting/`, `pedigree_evidence/`, `pedigree_scores/`, `pedigree/`,
`family_phase/` and `recombination/`. Simulation additionally creates
`founder_templates/` and `simulated_reads/`. Dataset names belong to the
run directory, not to stage identifiers.

This naming refactor also renames internal APIs, serialized classes and
identity keys. Historical checkpoint trees remain intact but are not accepted
as the new format. Use a fresh output root; preserve the frozen source with
older runs when reproducing them. Do not rename folders or edit identity
sidecars to force compatibility. Curated real-data handoffs retain their
original filenames and provenance.

## Simulation stage boundaries and chromosome shards

`simulate --stop-after-stage block_discovery` stops after block discovery;
`--stop-after-stage painting` stops after local feedback selection, final
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
  --process-contigs chr1 chr2 --stop-after-stage painting
```

Specify a stop boundary for shard workers, before genome-wide pedigree/family
inference. Shards retain chromosome checkpoints but do not publish global stage
completion. After every shard finishes, run once without shard or stop controls
to validate/reuse the chromosome outputs and continue genome-wide inference.

The TOML equivalents are `[run].process_contigs = ["chr1", "chr2"]` and
`[run].stop_after_stage = "painting"`. Existing environment controls
`BHD_SIM_CONTIGS=chr1,chr2` and `BHD_SIM_STOP_AFTER_STAGE=painting` also work.
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

All four reconstruction commands default to `--feedback-selection path`:

1. Preserve the original missing-aware 200-SNP discovery panels.
2. Fit and select those local panels under the normalized diploid path model,
   before any context assembly (**initial local fitting**).
3. Assemble the selected panels through L1, split/refit the context back to
   200-SNP candidates, and select locally again.
4. Assemble the latest selected panels through L1+L2, split/refit again, and
   select locally again using original and feedback-derived candidates.
5. Optionally perform one same-count segment-exchange pass.
6. Proceed to final L1–L4 assembly, founder refinement and downstream inference.

Steps 3–4 are the two feedback/refinement rounds. There is no additional
feedback stage after them. Initial fitting is local model fitting, not another
assembly round. Selection follows **each** feedback round.

The normalized path model sums over diploid copying paths, includes an explicit
unknown state, learns founder frequencies and penalizes dictionary complexity.
BIC/cavity fits construct starts and candidates; the normalized regularized path
likelihood selects the final panel. A previous better-scoring panel is retained
under the unchanged observations/objective. Released alleles require conditional
one-bit support plus directional carrier evidence; these are not full Bayesian
allele posterior probabilities. Missing calls remain `-1`.

Assembly supplies candidate sequences, not extra reads. The context carrier
model excludes the focal block's emission before its alleles are refitted.
Truth and downstream painting/L4 accuracy never select local panels. The same
calibrated genotype likelihoods and observation masks feed initial fitting,
both feedback rounds, final assembly and downstream inference. Supplied maps
and the configured generation multiplier determine path transitions; missing
maps use the configured fallback rate.

The optional final pass exchanges suffixes between two local rows, screens
all pairs at up to 16 cuts and refits the top eight plus an ordinary warm-start
control. It retains the original panel if fitting worsens the objective and
cannot directly change founder count. Enable it with
`--feedback-segment-exchange`; disable it with
`--no-feedback-segment-exchange` (the default).

`--feedback-selection balanced` and `strict` retain the explicit cavity-rescue
alternatives. They select after L1 and L1+L2 but omit the initial path fit.
Balanced allows broader local rescue/refinement; strict protects clean backbone
calls and rescues less variation. Segment exchange requires path selection.

TOML uses `[run].feedback_selection = "path"` and
`[run].feedback_segment_exchange = false`. Environment equivalents are
`HAPLOTYPES_FEEDBACK_SELECTION` and `HAPLOTYPES_FEEDBACK_SEGMENT_EXCHANGE`.
Precedence is CLI > TOML > environment > defaults. These are local block
operations, **not** family-phase or pedigree feedback.

Exact context-site inference is bounded to ten distinct context haplotypes;
larger or partially covered contexts retain their local proposals and record a
skip reason. This is a computational cap on context proposals, not a biological
founder-count cap. Path selection does not impose K=8 or K=10.

The optimized full TroMau chr1 prototype took 56.32 minutes on 76 cores for
initial fitting plus both assembly/selection rounds, or 57.68 minutes with the
optional segment pass. This excludes raw discovery, calibration estimation and
final L1–L4/downstream work. In two matched 100-block simulation controls, most
improvement over raw discovery came from initial fitting; feedback added two
exact local rows in the stressed control. Rare weak rows can still be lost;
real-data counts are not independently established founder truth.

Raw discovery is never overwritten. Initial fitting, each feedback round and
optional exchange have separate resumable checkpoints. Toggling final-only
segment exchange can reuse core rounds. Changed read-model calibration makes
old likelihood/discovery checkpoints scientifically incompatible: preserve old
runs and use a new output/checkpoint root; do not relabel old products.

## Final founder refinement

`--founder-refinement on` is the default for all reconstruction commands.
After each executed L1, L2, L3 and L4 level of **final assembly**, the complete
refiner reopens original prepared local-row choices inside the current component
boundaries. Its result feeds the next hierarchy level. Bounded deletion/refitting
can reduce the initial founder count under the existing complexity cost.
The existing early stop for an irreducible hierarchy is retained; unused levels
do not cause extra refinement passes.

Initial local fitting and the two L1/L2 context-feedback passes never run this
final-assembly refiner, whether selection uses path, balanced or strict.
Use `--founder-refinement off` for a controlled
comparison without any final-assembly refinement. Toggling this option within
one code version reuses local feedback checkpoints; only final assembly/painting
and downstream products change identity.

TOML uses `[run].founder_refinement = "on"` or `"off"`; the environment variable
is `HAPLOTYPES_FOUNDER_REFINEMENT`. Precedence is CLI > TOML > environment > on.
The API exposes `AssemblyConfig.founder_refinement_config`. Its default beam
budget grows from 64 to at most 1024 only when the data-based search-gain rule
warrants it; it is not a confidence cutoff. When this local search stalls, final
L1 pieces, both original and refined, supply larger proposals at the initial
beam width. These must improve the primary score, or the partial-data score at
an exact primary tie, without worsening primary genotype fit. This multiscale
fallback is
part of the default refiner; no extra flag is needed. See
[methods](methods.md#final-founder-path-refinement).

The final refinement also compares bounded one-founder deletions after refitting
under the existing complexity cost, completed exact-flank window searches,
and paired suffix/interval exchanges. These use no generation labels or known
founder count; the off flag disables the entire final refiner. Count refits can
be expensive on difficult chromosomes. See the measured limits in
[validation](validation.md#expanded-founder-search-at-n80-and-5x).

Each level saves its `refinement_l1` through `refinement_l4` result beneath
`assembly/`. Independent small components save compact completed
paths; long components also retain detailed beams, iterations and proposals.
A completed component can be reused when a sibling is interrupted. The
`founder_refinement` aggregate remains the final-product checkpoint.

Components share read-only chromosome evidence and divide the existing core
budget; surviving components and focal searches acquire freed threads at
numerical boundaries. The refiner does not overlap with the painting pool.
At L1/L2, a chromosome-wide minimum proposal-bin size avoids giving every small
component a full 2,000-bin search budget. L3/L4 retain component-specific proposal
resolution. This changes candidate exploration, not full-site acceptance.
Changed code/configuration is part of assembly and downstream cache identities;
retain older accepted outputs and
use a separate run root. Raw reads and original Block discovery need not be
regenerated. These are assembly checkpoints, not new globally complete stages.

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
not block discovery or downstream models.

For Python callers, `AssemblyConfig.panel_search_config=PanelSearchConfig(...)`
selects bounded search. Its controls are `paths_per_endpoint` (default 16),
`max_sweeps` (100), `full_scores_per_kind` (16), `max_bins` (2000) and
`tensor_budget_mb` (256). These are not one-to-one replacements for broad
search's controls. Initial panels from every input anchor compete under the
same full Viterbi/BIC score. Single and coordinated deletions are proposed
using conditional reassignment costs, then accepted only after full repainting.
Search stops early when no tested edit improves the objective; reaching 100
sweeps is reported as budget-limited, not as convergence.

The `AssemblyConfig` fields `beam_width`, `max_founders`, `top_n_swap`,
`max_cr_iterations`, `paint_penalty` and `min_hotspot_samples` apply **only to
broad search**, selected by `panel_search_config=None`. Their stored defaults
are inactive placeholders in bounded mode; changing them in that mode raises
an error instead of silently doing nothing. In particular, bounded search does
not use `max_founders` as a fixed founder-count cap. Shared settings such as
`cc_scale` and the transition model remain applicable to both searches.

Structured transitions are a restricted statistical model, not an exact
acceleration of arbitrary dense transitions. Read the
[scaling and accuracy trade-offs](founder_scaling.md).

Assembly selection does **not** change block discovery. Its established search remains
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
| `founder_templates/` | Frozen simulation sequence inputs |
| `simulated_reads/` | Simulated observations, true pedigree, alleles and raw crossover events |
| `block_discovery/` | Discovered 200-SNP block haplotypes, raw likelihoods/observation masks as applicable |
| `feedback_path_initial/` | Initial local path fits, per-block checkpoints and selected panels |
| `feedback_path_l1_assembly/`, `feedback_path_l1/` | L1 context, local proposals, per-block path fits and selected panels |
| `feedback_path_l2_assembly/`, `feedback_path_l2/` | L1+L2 context from selected L1 panels, proposals and selected local panels |
| `feedback_path_exchange/` | Optional final same-count segment proposals and selected panels |
| `feedback_balanced_*`, `feedback_strict_*`, `feedback_l1*` | Separate checkpoints for explicitly selected cavity-rescue alternatives |
| `genotype_evidence/` | Lossless compact GL/position/observation-mask cache for downstream inference |
| `assembly/` | Preprocessing, L1–L4 levels, per-level founder-refinement components/searches and final aggregate |
| `painting/` | Typed component-local painting products |
| `painting/reports/CONTIG/` | Founder callability/carrier support and assembly-search TSVs |
| `pedigree_evidence/versions/ID/` | Reusable prepared chromosome evidence |
| `pedigree_scores/versions/ID/` | Compact genetic likelihoods, candidate panel, per-chromosome resume products |
| `pedigree/versions/ID/` | Versioned pedigree decisions and support tables |
| `pedigree/_global.p5.b2` | Current published pedigree view consumed downstream |
| `family_phase/` | Final phase products, `CONTIG.iterations.p5.b2` work checkpoints, and compact completion summary |
| `recombination/` | Conditional map products and completion summary |

Atomic protocol-5/Blosc checkpoints use `.p5.b2`; completion markers are written
only after required outputs exist. Inputs, model/configuration and relevant
source identities are checked at stage boundaries. Interrupted work resumes
from durable chromosome, assembly-phase or family-iteration checkpoints. Source
changes may invalidate affected stages; do not bypass a mismatch by relabeling
an old checkpoint as current.

Reconstruction writes compact genotype evidence after validating its inputs.
Pedigree and family phase use this derived cache when it matches the source checkpoint files;
older runs without it continue to read their original raw/block checkpoints.
The cache does not replace the rich simulation truth or discovery products:
keep those source files, including linked targets, for validation and resume.
It uses the same likelihood precision and missing-observation mask, trading an
additional write and disk space for smaller repeated reads and lower memory.

Family phase uses one phase-focused path, with no separate full-posterior or imputed
source stage. Iteration checkpoints retain family messages, the preceding
polished phase and the consecutive-stability count; large numerical caches are
rebuilt on restart. A 520-iteration safety limit refuses release if phase has
not stabilized. Stable phase does not assert marginal-posterior convergence.

Family phase preserves published genotype calls and missingness: family-supported fills
may inform polishing internally, but are not released as newly imputed alleles.
The retired `--impute-missing` flag and TOML `[refinement].impute_missing` key
are rejected. Remove the TOML key (even if set to `false`); there is no replacement
switch for publishing the former separate imputed product.

Changes confined to family phase can reuse compatible painting/pedigree/raw checkpoints in a new
downstream output root. The local-feedback change described above also changes
painting inputs: reuse only compatible original input/discovery stages for that
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

## Recording a publication run

Retain the exact source commit, the command and selected TOML configuration,
the input sequences/variant files and their sample order, the generating seed
and map (for simulations), and the inference settings. Archive the environment
versions and readable evaluation tables alongside the run's checkpoints.
A seed alone is not sufficient to reproduce a run with different founder
templates, source code or model settings. Published data access and citation
details must identify the actual inputs; the local example paths are not
public download locations.

The checkout checks on 14 September 2026 used Python 3.12.0 on Linux with the
following installed versions. This is a tested environment snapshot, not an
exhaustive dependency lock or validation of every version permitted by
`pyproject.toml`.

| Dependencies | Tested versions |
| --- | --- |
| NumPy / Numba / SciPy | 2.4.0 / 0.64.0 / 1.16.3 |
| pandas / hdbscan / cyvcf2 | 2.3.3 / 0.8.41 / 0.32.1 |
| Blosc2 / TBB | 4.5.1 / 2022.3.1 |
| Matplotlib / NetworkX | 3.10.8 / 3.6.1 |
| tqdm / openpyxl / setuptools | 4.67.1 / 3.1.5 / 80.9.0 |

## Pedigree cache reuse and explanations

Pedigree now has three independently identified products:

1. Preparation depends on painting/raw inputs, sample/contig order, binning/source
   settings, and preparation code—not direction or release thresholds.
2. Genetic scoring also depends on likelihood settings, actual candidate
   eligibility and the fixed pair-screen policy. It checkpoints each scored
   chromosome and a compact aggregate; large source-factor arrays are omitted.
3. Decisions depend on the score identity, all decision settings, caller
   chronology and decision code, including bootstrap and graph policy.

Changing direction/family settings or support thresholds through the API
reuses preparation and scores; changing candidate eligibility reuses
preparation but rescores. A score-stage change never masquerades as a
decision-only replay. Ordinary atomic replacement of an upstream checkpoint
changes its source-file identity. pedigree v5 does not retroactively relabel older
v4 preparation caches; the first new run prepares them once.

Versioned decisions are retained. The conventional pedigree global file is the
current published view; a pre-existing different view is archived before
replacement. Do not run competing global publishers in the same output root.
Family phase already keys conditioning on the actual Tier-B relationship table, so
unchanged relationships reuse final phase. If relationships change, family phase
rejects incompatible existing checkpoints: preserve those results and use a
separate downstream output/checkpoint root rather than deleting or relabeling
them. This change does not add an automatic family phase/recombination rerun or feedback loop.

The standard `pedigree/` exports now include:

- `call_explanations.csv`: one row per sample, its release/ambiguity status,
  candidate sets, state/identity margins, stability and graph intervention.
- `alternative_explanations.csv`: the local and graph selections plus up to
  three leading scored identities **within each parent-count state**. Genetic
  log evidence (after the existing contamination mixture), structural state
  treatment, direction, reciprocal-family and joint-ancestry contributions
  are separate. Priors/multiplicity, marginal state evidence and graph utility
  are also separate: they are not all additive terms of a single posterior.
- `tier_b_candidate_sets.csv`: primary ambiguity sets, without inventing
  a second known parent where only a parent-count state is supported.
- `search_diagnostics.csv`: M2 panel size, omitted parent count, screen ranks,
  boundary score gap and whether a released parent is at the screen boundary.
  M1 continues to score every eligible parent.

These are explanations of the fitted composite model, not calibrated
probabilities, proof of biological direction, or validation against truth.

## Observed-reference consistency

The Tropheops workflow writes `reference_consistency.csv` separately from
pedigree and known-truth simulation outputs. It compares discovered local
haplotypes with confident G0 genotype calls, including a best-pair dosage
comparison at heterozygous reference sites.

The table reports `n_g0_reference_samples`, `references_matched_pair`,
`references_matched_hom_under_2pct`, `haplotypes_matching_reference` and
`haplotypes_without_reference_match`, alongside per-reference errors and
evaluated-site counts. These are compatibility summaries, not counts of true
founders or demonstrated chimeras. The available references may not represent
all ancestral sequence, and a matching pair is not necessarily uniquely phased.
When reference samples enter discovery, the comparison is non-independent.

## Founder support and bounded-search diagnostics

Every completed or resumed painting chromosome writes small reports under
`checkpoints/painting/reports/CONTIG/`:

- `founder_support.tsv`: called/missing marker alleles per component-local
  founder, component boundaries, distinct observed carrier samples, and called
  sites without a released named carrier having positive read depth.
- `founder_support_windows.tsv`: the same callability/carrier summaries in
  at most 200-marker windows. Coordinates name the first and last variants,
  both **1-based inclusive**; these are not BED coordinates.
- `assembly_search.tsv`: per-level/per-panel sweep budgets, truncated proposal
  categories, improving proposals at the last scored rank, iteration-cap hits,
  and the two leading evaluated alternatives with their BIC-like score gap.
  A gap of at most two score units is labelled close purely for inspection;
  it is not a release threshold or a calibrated Bayes-factor claim.

Carrier counts require both the public named painting and observed depth.
A homozygous named carrier counts once per sample/site. Unknown or pooled
ancestry is not attributed to an arbitrary founder. Missing founder alleles
remain missing and get no called-allele support. Related carrier samples are
not independent ancestral lineages; a called allele without a direct named
carrier is not automatically wrong. No additional masking is applied.

Assembly alternatives describe the **particular hierarchy sweep before
subsequent founder refinement**, not a posterior over final chromosome paths.
Both genotype/model and search uncertainty may remain. A small scored gap,
an accepted boundary proposal or a reached budget is a reason to inspect a
region, not proof that broader search will improve it. Conversely, a large
gap among evaluated proposals cannot certify ungenerated or truncated
alternatives. The reports never automatically expand the search. Existing
`--assembly-search broad` remains an explicit controlled comparison.
