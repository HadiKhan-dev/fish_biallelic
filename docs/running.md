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
full-node utilization. Contig reconstruction (feedback, assembly, painting),
pedigree evidence/scoring, family phase, recombination and saved-result
evaluation use chromosome-parallel workers within an allocation, subject to
their memory estimates (`--contig-workers`, below, for reconstruction). Surviving
workers take freed threads at numerical (reconstruction: contig) boundaries;
genome-wide pedigree decisions remain barriers.

Package startup isolates compiled Numba caches by run/job, host and code
version, while workers within that scope share them; workflows keep compiled
kernels between runs in `~/.cache/haplotype-reconstruction/numba` (or under
`$XDG_CACHE_HOME`) and compile a new code version's kernels in parallel before
their first pool. No launch-wrapper setting is required. Scientific checkpoints
remain independent and persistent. See
[native compilation caches](performance.md#native-compilation-caches) for
the store and the explicit cache override.

## Optional cross-node batches

The ordinary commands need only one allocation. To let other allocated nodes
help with independent local-refit and L1–L4 hierarchy batches, add a shared
queue directory:

```bash
# Coordinator, on its own allocation.
haplotypes reconstruct --vcf cross.bcf --output work/runs/my_cross \
  --contigs chr1 chr2 chr3 --cores 76 \
  --batch-queue /shared/my_cross/batches

# Helper, launched separately inside another allocation.
haplotypes batch-worker --queue /shared/my_cross/batches --cores 48
```

The same `--batch-queue` option is available for `simulate`, `astcal` and
`tropheops`; TOML uses `[run].batch_queue`. An explicit CLI path overrides
TOML, then `HAPLOTYPES_BATCH_QUEUE`. Omit all three for node-local execution.

Both processes must have the same installed/frozen package source and access
to the queue directory. Set each `--cores` within that process's actual
allocation. The package does not submit allocations, reserve extra CPUs or
move a shell onto another node. Multiple helpers can join the same queue.
They exit after 120 seconds without work by default
(`--idle-seconds` changes this), so launch them while batches are available.
A coordinator also executes work; it is not an idle dispatcher.

Only independent local fitting and hierarchy batches use this transport.
Discovery, coupled founder refinement and downstream stages do not become
cross-node jobs by enabling it. Each keeps its existing local scheduling.
Results retain their task IDs and scientific batch boundaries; only the
coordinator writes the normal pipeline checkpoints. A single hierarchy batch
stays local. See [execution ownership](development.md#optional-work-sharing-between-allocations).

Keep the queue and source snapshot while resuming interrupted work. Confirm
the previous coordinator is terminal before restarting it; do not run two
owners of the same scientific checkpoint stage. Expired Slurm coordinators
stop offering work, and expired worker claims can be reclaimed. Completed
hierarchy bundles can be reused under the exact checkpoint identity.
For changed source, use a fresh queue: do not relabel old results to force reuse.
Retained transport files can be removed once the run and any desired resume
are finished, independently of the scientific checkpoints.

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
| `--founder-refinement on\|off` | on | Progressive final L1–L4 refinement with converged founder-count decisions |
| `--contig-workers auto\|N` | auto | Contigs reconstructed concurrently within `--cores` (feedback, assembly, painting); auto is one per eight cores, memory-limited; 1 is sequential. Execution only |
| `--shared-family-evidence` / `--no-shared-family-evidence` | on | Shared phase-error evidence in downstream recombination maps |
| `--order-breakpoints FILE` | none | Reference-order breakpoint TSV for the recombination map (see [Reference-order breakpoints](#reference-order-breakpoints)) |

For these switches, explicit CLI values override TOML, then the supported
environment settings, then defaults. These TOML keys belong in `[run]`, except
`shared_family_evidence` and `order_breakpoints`, which belong in `[recombination]`. Boolean settings
accept `true/false`, `on/off`, `yes/no` and `1/0` strings, or TOML booleans.
Recombination also exposes its shared-family switch and breakpoint file in the standalone command.
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

The self-calibrated site-quality filter runs before read calibration; it is on
by default for VCF/BCF input and off for simulations. Override either default
with `--site-quality-filter` / `--no-site-quality-filter` (TOML:
`[run].site_quality_filter`, environment `HAPLOTYPES_SITE_QUALITY_FILTER`).
Its setting is part of the block discovery identity, so filtered and
unfiltered checkpoints never mix. See [methods](methods.md#site-quality).

The per-sample contamination QC is decided per contig inside the site filter
and is on by default wherever the filter runs (VCF/BCF input), off for
simulations; override with `--sample-quality-filter` /
`--no-sample-quality-filter` (TOML: `[run].sample_quality_filter`, environment
`HAPLOTYPES_SAMPLE_QUALITY_FILTER`). Flagged samples are listed in the log, in
`OUTPUT/sample_quality/<contig>.tsv` and in the run record; inherited
heterozygous segments (reported, never flagged) are in
`OUTPUT/sample_quality/<contig>.segments.tsv`. The setting is part of the block
discovery identity. See [methods](methods.md#sample-quality).

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
`allele_refit/`, `family_phase/` and `recombination/`. Simulation additionally creates
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

## Reference-order breakpoints

Every stage assumes that reference order is meiotic order. Where a misplaced,
misassembled or inverted reference segment breaks that assumption, about half of
all meioses change transmitted homologue at the junction, and the map reports
these real transmission changes as a crossover pile-up. An optional breakpoint
file tells the recombination stage where reference adjacency is not genetic
adjacency:

```bash
python run.py recombination --output work/runs/tropheops_breakpoints \
  --checkpoints CHECKPOINT_ROOT --order-breakpoints order_breakpoints.tsv
```

The TOML equivalent is `[recombination].order_breakpoints = "FILE"`; the
option is accepted by every command that reaches the recombination stage.
Precedence is CLI > TOML > none, like `--recombination-map`: the CLI sets
`BHD_RECOMBINATION_ORDER_BREAKPOINTS` from them and clears it otherwise, and the
run record lists it. Direct Python callers can pass
`run_recombination(order_breakpoints=FILE)` / `run_from_checkpoints(...)` or set
that environment variable. There is no default file: a list of breakpoints is
reference-structure curation, not something the pipeline infers.

The file is tab-separated with a header line `chrom start end reason`;
`#` lines are comments. Positions are 1-based inclusive bp, and the order
discontinuity lies somewhere inside `[start, end]` (for example the contig gap
at the junction). A row whose reason starts with `reversed_segment` declares an
inverted reference segment instead: both of its ends act as breakpoints and calls
inside it are flagged `in_reversed_segment` (they stay in reference
coordinates). The stage widens every interval to the painting bins it overlaps,
removes those sites from the map axis and splits the component there, so no
call, exposure or shared-orientation tract crosses a supplied breakpoint. The
file path and its SHA-256 enter the recombination stage identity, and nothing
upstream depends on the file. A checkpoint root whose recombination stage was
made with another file (or none) refuses the new one; map with a different file
from a checkpoint root holding only the upstream stages (for example linked
copies), which recomputes recombination alone. The same format serves family phase's protected intervals
(`BHD_FAMILY_PHASE_PROTECTED_INTERVALS`), which only stop root-frame repairs and
do not mask anything.

Without a file the recombination stage still reports a truth-free junction
detector table (`junctions.csv`, see the outputs below); it never masks. Use it
to find candidate loci for curation, not as a breakpoint list by itself.

## Local feedback selection

All four reconstruction commands use normalized path selection:

1. Preserve the original missing-aware 200-SNP discovery panels.
2. Fit and select those local panels under the normalized diploid path model,
   before any context assembly (**initial local fitting**).
3. Assemble the selected panels through L1, split/refit the context back to
   200-SNP candidates, and select locally again.
4. Assemble the latest selected panels through L1+L2, split/refit again, and
   select locally again using original and feedback-derived candidates.
5. Proceed to final L1–L4 assembly, founder refinement and downstream inference.

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

These are local block operations, **not** family-phase or pedigree feedback.

Removed options. The structured transition model and the broad assembly search
were removed; the batched discovery search, the balanced/strict feedback
selections and the feedback segment exchange can no longer be selected. Their
CLI flags no longer exist. A TOML
key (`assembly_model`, `assembly_search`, `discovery_search`,
`feedback_selection`, `feedback_segment_exchange`) or `HAPLOTYPES_*` environment
variable set to the remaining value (`dense`, `bounded`, `standard`, `path`,
false) is ignored with one warning; any other value stops the run with a message
saying what was removed. Delete these settings.

Context-site inference is exact for at most ten distinct context haplotypes
and uses the polynomial uncertain-core site posterior above that (see
`methods.md`), so no context component is skipped for its founder count;
partially covered contexts retain their local proposals and record a skip
reason. The per-component feedback diagnostics count sites by method (exact,
core cube, core ball), core sizes, belief-propagation non-convergence, ascent and
recentring backstop hits, complement-start optima, multimodal abstentions and
shell-guard abstentions; the `FEEDBACK site posterior` log line prints their
totals (the assembly preprocess prints `[Preprocess] site posterior` when any
site has more than ten unresolved founders). Path selection does not impose K=8
or K=10.

The optimized full TroMau chr1 prototype took 56.32 minutes on 76 cores for
initial fitting plus both assembly/selection rounds, or 57.68 minutes with the
optional segment pass. This excludes raw discovery, calibration estimation and
final L1–L4/downstream work. In two matched 100-block simulation controls, most
improvement over raw discovery came from initial fitting; feedback added two
exact local rows in the stressed control. Rare weak rows can still be lost;
real-data counts are not independently established founder truth.

Raw discovery is never overwritten. Initial fitting and each feedback round
have separate resumable checkpoints. Changed read-model calibration makes
old likelihood/discovery checkpoints scientifically incompatible: preserve old
runs and use a new output/checkpoint root; do not relabel old products.

## Final founder refinement

`--founder-refinement on` is the default for all reconstruction commands.
After each executed L1, L2, L3 and L4 level of **final assembly**, the complete
refiner reopens original prepared local-row choices inside the current component
boundaries and fits each component to a converged local optimum of the full-site
objective. Its result feeds the next hierarchy level. Founder-count decisions
compare converged fits under the join's effective-carrier rule on the total gain
and its sequence part (scored exactly, without the join's near-identity
shortcut): every level can delete rows (rows without a private cell first) and
the last level can also add them.
The existing early stop for an irreducible hierarchy is retained; unused levels
do not cause extra refinement passes.

Initial local fitting and the two L1/L2 context-feedback passes never run this
final-assembly refiner.
Use `--founder-refinement off` for a controlled
comparison without any final-assembly refinement. Toggling this option within
one code version reuses local feedback checkpoints; only final assembly/painting
and downstream products change identity.

TOML uses `[run].founder_refinement = "on"` or `"off"`; the environment variable
is `HAPLOTYPES_FOUNDER_REFINEMENT`. Precedence is CLI > TOML > environment > on.
The API exposes `AssemblyConfig.founder_refinement_config`. The search and
count rules are fixed: converged Gauss-Seidel sweeps of exact window proposals
and founder-pair suffix exchanges, with final L1 pieces, both original and
refined, as larger moves; effective-carrier count decisions on the total gain
and its sequence part between converged fits; the best-F converged panel at the
decided count; and the smaller count
with its better-F panel after a count cycle. No option, environment variable or
TOML key selects another founder search, count criterion or panel rule. See
[methods](methods.md#final-founder-path-refinement).

Every count decision refits each one-row deletion (and, at the last level, each
duplicated-row addition seed) to convergence, so a decision costs about K
converged fits and can be expensive on difficult chromosomes. The decisions use
no generation labels or known founder count; the off flag disables the entire
final refiner. Each level prints its fits at the sweep backstop, count
backstops, count cycles, ambiguous count calls and best-F panel replacements;
the refinement diagnostics record each decision and replacement.

Each level saves its `refinement_l1` through `refinement_l4` result beneath
`assembly/`. Independent small components save compact completed
paths; long components also retain their converged fits.
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

## Assembly search

Every assembly level uses one route: learned dense linker transitions, one
dense transition mesh per batch, an endpoint-quota beam and the certified
candidate-panel search, with the 20-iteration linker limit. There is no
transition-model or search-breadth option and no automatic founder-count
cutoff.

The panel search starts from the initial panel of largest join likelihood
among those of every input anchor. Same-count edits are accepted when they
raise the join likelihood and leave every changed path a private cell (and
more than 1% of called sites from every other path). Tier 1 is the bounded
sweep: proposal statistics rank edits and the best improving panel is accepted
after an exact score. At each tier-1 stall the count step decides one founder
count change with the effective-carrier rule on the total and sequence parts
of each sample's gain (deaths: paths without sequence of their own first, then
the best-F candidate; births: the best accepted pool path; duplicate-seed
births at certified panels). When the count step is quiescent, tier 2 bounds
every same-count move of an explicit neighbourhood (replacement by any pool
path, any one-block row change, suffix copies and exchanges at every
boundary), scores exactly the moves the bound cannot exclude, and accepts the
best improving one. Before a panel is reported as certified, the endpoint beam
is re-run with doubled quotas (32 up to 256 paths per endpoint, times the
number of panel paths sharing an endpoint) and replacements (likelihood) and
births (count rule) by the new paths are tested the same way; a move found at a
level raises the quota to that level, and a panel is certified only when no level
up to the ladder's top adds one. The top is 256 for every join; a join whose
256 level accepted a move climbs on to 512 and 1024. Two reported backstops
(hits counted and printed, target 0): total sweeps <= 100 + 20 x (largest block
row count), and a move accepted at quota 1024. Per super-block diagnostics record tier-2 passes, survivors, exact scores,
pool sizes, the `certified`, `backstop_reached`, `ladder_extended` and
`pool_backstop` flags and every count decision (kind, margins on the total and
sequence parts, band, ambiguity); each hierarchy level prints their counts.

For Python callers, `AssemblyConfig.panel_search_config=PanelSearchConfig(...)`
holds `paths_per_endpoint` (default 16, the base endpoint quota),
`full_scores_per_kind` (16 tier-1 exact scores per proposal kind), `max_bins`
(2000) and `tensor_budget_mb` (256, the painting's back-pointer memory per sample chunk).

Assembly selection does **not** change block discovery, which keeps its
established search.

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
| `feedback_path_initial/` | Initial local path fits (checkpointed in chunks of 64 blocks) and selected panels |
| `feedback_path_l1_assembly/`, `feedback_path_l1/` | L1 context, local proposals, path fits (chunks of 64 blocks) and selected panels |
| `feedback_path_l2_assembly/`, `feedback_path_l2/` | L1+L2 context from selected L1 panels, proposals and selected local panels |
| `genotype_evidence/` | Lossless compact GL/position/observation-mask cache for downstream inference |
| `assembly/` | Preprocessing, L1–L4 levels, per-level founder-refinement components/searches and final aggregate |
| `painting/` | Typed component-local painting products |
| `painting/reports/CONTIG/` | Founder callability/carrier support, per-sample painted switches and assembly-search TSVs |
| `pedigree_evidence/versions/ID/` | Reusable prepared chromosome evidence |
| `pedigree_scores/versions/ID/` | Compact genetic likelihoods, candidate panel, per-chromosome resume products |
| `pedigree/versions/ID/` | Versioned pedigree decisions and support tables |
| `pedigree/_global.p5.b2` | Current published pedigree view consumed downstream |
| `allele_refit/` | Post-painting founder allele refit per chromosome (released and family-phase inference alleles) |
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
are neither polishing evidence nor released as newly imputed alleles.
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
In `<contig>.crossovers.csv`, `left_bp`/`right_bp` are the decoder's flanking
supported markers; `reported_left_bp`/`reported_right_bp` (inclusive) widen a
call whose flanks straddle a painting-bin edge to the two bins around that edge,
the painting's actual resolution (a change inside either bin is placed at the
edge); `in_reversed_segment` marks calls inside a supplied reversed segment.
Counts and maps use the calls themselves, not the reported widths.
`<contig>.junctions.csv` and the genome-wide `junctions.csv` list candidate
reference-order junction loci (calls of at least three parents chained by
overlapping reported intervals) with per-parent odd/decidable meiosis counts,
the mixing and pairing log likelihood ratios, `junction_log_lr`, `flagged`
(at least ln 100) and an interpretation (`junction`, `parent_frame_flips` or
`recombination_or_unresolved`); [methods](methods.md#family-refinement-and-recombination)
describes the model. `summary.csv` adds `junction_loci`, `junction_loci_flagged`,
`order_breakpoints` and `order_breakpoint_masked_sites`.
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

Three further views sit beside those columns. `long_phase_switches_min5` and
`_min20` count orientation changes between runs of correct-genotype true
heterozygotes, bridging uncalled or wrong ones, when both runs hold at least 5
or 20 heterozygotes; single-site flips, which dominate `phase_switch_errors`,
are excluded, and a bad call at the switch point does not hide a switch.
`segment_allele_errors` relabels each founder row segment-wise (a truth-label
switch costs 20 mismatches); `chimeric_segment_charges` is the rest of
`called_allele_errors`, charged because a row follows another founder for a
segment, and `segment_label_switches` counts those segments.
`called_interval_hit_rate` is the fraction of called crossover intervals on
correct edges that contain a true crossover (a localization check).
`reported_interval_hit_rate` applies the same rule to the reported intervals;
read it with `called_interval_median_width_bp` and
`reported_interval_median_width_bp`, since a wider interval contains more by
construction. These columns follow all earlier ones.

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
  Pooled candidates are listed separately: `ObservedPooledCandidateSamples`,
  `MeanObservedPooledCandidates` and
  `CalledSitesWithoutObservedNamedOrPooledCandidate` count a row wherever it
  is a candidate of a pooled painted track (rows identical in that bin, named
  by the chunk's `hap1_members`/`hap2_members`) with positive read depth.
- `founder_support_windows.tsv`: the same callability/carrier summaries in
  at most 200-marker windows. Coordinates name the first and last variants,
  both **1-based inclusive**; these are not BED coordinates.
- `assembly_search.tsv`: per-level/per-panel sweep budgets, truncated proposal
  categories, improving proposals at the last scored rank, iteration-cap hits,
  the two leading evaluated alternatives with their join log-likelihoods,
  and the join's count decisions (`count_death`, `count_birth`, `count_seed`).
  These are for inspection; they are not calibrated Bayes-factor claims.
- `founder_counts.tsv`: one row per component: `FounderRows`, and how firmly
  that count is determined. `FounderCountAmbiguous` is the final count step's
  call (the last level's converged count pass, which decides the reported
  count): true when one of its decisions has a bootstrap band containing zero,
  when it cycled, or when a restored best-F panel fails its own deletion test;
  `CountStepDecisions` lists the decisions (kind, founder count, outcome,
  margin, band). A count cycle (a row added by a search and deleted again) keeps
  the deletion; it is undecided when that deletion is ambiguous and the cycle
  began with a decided deletion. `JoinCountCycles`, `JoinUndecidedCycles` and
  `JoinAmbiguousCountCalls` count the final assembly's joins over the component,
  `FeedbackJoin*` the block-feedback context assemblies' joins, and
  `UndecidedCountCycles` all undecided cycles. The log names components with an
  ambiguous final count or undecided cycles. Diagnostics only; empty values mean
  the run predates these records.
- `sample_switches.tsv`: per sample, the painter's biological switch count
  (minimum homologue changes of the Viterbi path, summed over components),
  the cohort median over painted samples (not split by generation) and the
  ratio to it. The log names the three highest samples. A sample far above
  the median is not explained as a founder mosaic by this panel (for example
  foreign reads or a missing haplotype), or marks a panel error. It is a
  diagnostic only; no sample is filtered or down-weighted.

Carrier counts require both the public named painting and observed depth.
A homozygous named carrier counts once per sample/site. Unknown or pooled
ancestry is not attributed to an arbitrary founder; pooled candidates are
counted only in the separate pooled-candidate columns (once per sample/site). Missing founder alleles
remain missing and get no called-allele support. Related carrier samples are
not independent ancestral lineages; a called allele without a direct named
carrier is not automatically wrong. No additional masking is applied.

Assembly alternatives describe the **particular hierarchy sweep before
subsequent founder refinement**, not a posterior over final chromosome paths.
Both genotype/model and search uncertainty may remain. A small scored gap,
an accepted boundary proposal or a reached budget is a reason to inspect a
region, not proof that broader search will improve it. Conversely, a large
gap among evaluated proposals cannot certify ungenerated or truncated
alternatives. The reports never automatically expand the search.
