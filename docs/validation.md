# Validation and known limitations

The pipeline is evaluated with known-pedigree simulations and fixed-input
implementation comparisons. These answer different questions: numerical
equivalence does not establish biological accuracy, and a correct pedigree does
not imply perfect founder reconstruction or sample phase.

Default simulations use frozen empirical founder templates, 5× mean read depth
and 20/100/200 sample cohorts. The 22 template contigs are chr1–20, chr22 and chr23.
Seeds400–402 are development data; seed401 was held out specifically when
choosing the family phase stopping rule, not from all project development.

## Numerical, performance and product checks — 25 September 2026

Two implementation improvements retain the scientific models: fused read-model
likelihood/gradient and shared-binomial EM kernels, and an optimistic upper-bound
screen before losing final founder-count additions. Compared with the frozen
pre-change working tree:

- Analytic likelihoods/gradients matched the NumPy reference within 2.9e-14 and
  3.6e-15 respectively, including dispersion limits, pooled effects and missing
  training rows; finite-difference gradient checks also passed.
- All EM starts on the Tropheops and Astcal observed-AD histograms retained their
  stopping iterations. Both full nested fits selected the same model. An
  application fixture included missing observations and read depth 120, beyond
  both the fitting and lookup caps; there were no genotype-argmax changes.
- The founder-count relaxation bounded all 1,971 enumerated small panels.
  Nine cached regional/full-chromosome comparisons retained exactly the same
  ordered hard calls, selected local-row paths and completed-resume results.
  The accepted four-to-five-founder addition remained accepted.
- Four actual-read, 64-block controls (seed7000 chr3, 7004 chr10, 7019 chr7 and
  7021 chr3) fitted the model on their complete observed chromosome and reran
  discovery, both feedback rounds and regional final assembly. Every ordered
  hard call matched at discovery, selected feedback and final assembly.
  Their final called-error/call totals were respectively 0/76,800, 907/63,946,
  0/76,798 and 0/76,800 in both arms. These are regional founder results,
  not whole-chromosome accuracy estimates or sample-phase metrics.
- The new `founder_refinement` evaluation stage reads the final count-up
  product, falling back to the final founder-refinement checkpoint. Earlier
  hierarchy-level metrics retain their original meaning. A typed checkpoint
  fixture and an actual completed assembly verify that distinction.

There is one separate scientific correction: a sample with no training-fold
reads now receives the estimated pooled prior mean for error/balance, rather
than the shared-model initialization. This is the mode of its otherwise
uninformed normal-prior objective. Two artificial missing-training-row controls
changed some predictive scores and raw GLs, with no genotype-argmax changes in
the application fixture; an improvement in every held-out score is not claimed.
Read-model version v3 invalidates dependent old likelihood checkpoints without
deleting the saved data.

Family-sweep prototypes passed 36 overlapping-family fixtures and four saved
chromosome continuation checks; a 20-iteration, 26-branch continuation also
retained phase calls, factor messages, branch cavities and convergence deltas.
They were not promoted because the measured warm benefit was small.
These checks do not constitute a new complete-genome pedigree validation or
remove the previously documented calibration trade-offs.

All fitting uses observed reads only. Truth is opened for evaluation after
freezing the panels; ancestry-representation diagnostics use the actual sampled
truth painting, not an all-present assumption. Accepted source/results remain
untouched; comparison scripts and intermediate products are in
`.work/pipeline_polish_20260925.6STDyVgq/`, outside Git.

A fresh public-CLI smoke run (seed92501, N80, three synthetic chromosomes of
2,400 markers each) completed every stage, all-stage evaluation, BCF/TSV export
and complete resume on 112 CPUs. Exported founder and sample alleles exactly
match the checkpoints, including missing calls. This deliberately small example
released no pedigree edges and left all 80 individuals unresolved; it therefore
does not validate pedigree recovery or informative recombination estimation.
The saved-chromosome family checks above supply the nonempty-family coverage.
The wheel includes all 160 public Python modules and no manuscript, test,
checkpoint or private-work files.

## Predictive pedigree evidence calibration — 23–24 September 2026

Pooled predictive calibration is the default from 24 September, following
explicit approval of its aggregate benefit despite the known regressions below.
Use `--pedigree-calibration off` for the unscaled comparison. The estimator,
candidate universe, priors and release thresholds are unchanged from the
validated opt-in implementation; only default selection changed.
Nine fixed-upstream 22-chromosome decision replays, including fresh full-marker
Mendelian exclusions, improved exact parent configurations from **2,354/2,400
to 2,370/2,400**. Seed7010 improved 137→146/160 and seed7019 improved 137→144/160;
the other seven cohorts retained all 2,080 exact configurations. False edges
in partial Tier-B outputs fell 4→2, and missing true partial edges fell 29→15.
The all-M0 control retained 320/320 and fell back to evidence scale 1 because
there was no predictive calibration gain.

The aggregate improvement is not individual-level dominance: seed7019 gains ten
exact configurations but loses three previously exact calls to unresolved status
(F4_13, F5_2 and F10_26). Seed7010 gains nine and loses none. An unresolved call
is not counted as a wrong exact pedigree, but it is a loss of usable resolution.

Eight selected scales were 0.0167–0.0506, with positive untouched chromosome-test
gains. Estimation from cached scores took 0.37–1.46 seconds. The canonical pipeline
reused its genetic cache and reproduced the decision replay, exports and resume.
This is conditional predictive calibration of existing paintings and candidate
panels, not calibrated pedigree probabilities or independent validation of the
entire pipeline. Bootstrap/LOCO conditions on the fitted scale. No L1–L4 or final
phase rerun was part of this comparison. See [Methods](methods.md) for assumptions.

The promoted estimator pools three training/selection folds while preserving
a fourth test fold. Its nine full decision releases reproduce the same exact
configuration and partial-edge totals. Eight selected weights are 0.0150–0.0528;
the all-root cohort retains scale 1 at a boundary optimum. Release integration
reproduces all nine private fits exactly; testing-fold score changes do not alter
the fitted weight, and the disabled path returns the original evidence object.

Eight chromosome partitions per cohort test sensitivity, not confidence limits.
For seed7021, the fitted range narrows from 0.0141–0.0361 to 0.0186–0.0266.
Canonical releases at both new endpoints retain 320/320 (the original endpoints
gave 314–320). New endpoint ranges are 146–146/160 for seed7010 and 144–145/160
for seed7019. These development results support pooling but do not establish
well-calibrated posterior confidence. One previously identity-unresolved M1 count
on seed7019 becomes count-unresolved under the pooled fit.

Six subsequent conditional controls (7006, 7008, 7009, 7011, 7014 and 7023)
retain all **1,600/1,600** exact configurations and every true partial edge,
with no false edges, in both modes. They cover a bottleneck, final-generation-only
sampling, backcross/missing parents, deeper generations, random missing parents
and ordinary F1–F3 sampling. The pooled estimator was frozen before these tests.
Both arms reuse the same pre-optimization genetic scores and run current release
logic with fresh raw-marker exclusions; these are not new end-to-end runs.
Five fitted scales have positive untouched-test gains; the all-M0 cohort retains
scale 1 at a boundary optimum. Fits take 1.6–5.7 seconds on the nine-CPU runs.
Across all fifteen controls, exact configurations are 3,954→3,970/4,000; the
six additional controls contribute no gains or regressions to that difference.

Completing all 24 available complex-cross conditional controls changes this
assessment: exact configurations are **6,194→6,206/6,240**, false partial edges
4→3 and missing true partial edges 29→17. Seed7007 regresses 160→158, reversing
one parent–offspring edge; seed7018 regresses 160→158 through two abstentions.
The other seven newly added cohorts retain every exact configuration. Thus the
net gain is not a uniformly safer pedigree; default promotion explicitly accepts
this trade-off, rather than claiming improvement for every fish or cohort.
The fitted scale changes the balance between genetic evidence and the existing
finite direction/family factors. Better held-out read/likelihood prediction does
not by itself demonstrate more accurate biological relationship direction.

Subsequent fixed-input chr1 family-phase checks of the original predictive fit
are mixed. Seed7010 switches improve 162→158 on 8,290,043 common comparisons;
seed7019 worsens 193→209 on 8,010,397. Component-aligned allele errors improve
957,677→904,441 and 1,025,103→943,573 respectively, with unchanged genotypes.
The ordinary 7000/7021 controls are identical. These are four chromosomes, not
four whole-genome phase validations, and stable calls do not imply convergence
of every latent message.

The multi-model discovery candidate-bank remains a private experimental study.
Across 164 local 200-SNP blocks, a balanced candidate-bank rescue added 998 calls
without adding called mismatches or unmatched inferred rows. A much cheaper
visited-panel rescue improved retention further but added errors and extra rows
in rare-founder controls; it was **not promoted**. These local founder metrics
must not be interpreted as chromosome-assembly or painted-sample phase accuracy.

A subsequent four-region comparison (128 contiguous blocks per region) found
identical final regional assemblies with and without candidate-bank rescue:
the ordinary two-round feedback absorbed its initial local gains. One initial
block gained one wrong call. All four original-read fits selected the nested
binomial model. Controlled heterogeneous/overdispersed-read comparisons are
separate; the earlier 164-block result does not establish uniform safety.

The completed seed7019 whole-genome shared-read-calibration comparison also
shows a downstream trade-off, separate from pedigree evidence calibration.
On 1,287,925,179 common complete genotypes, genotype errors changed
2,479,722 → 2,507,900. On 195,669,226 common phase comparisons (correct genotype
in both arms and common component boundaries), phase switches changed
4,980 → 5,389. Better read-model predictive likelihood alone therefore does not
establish improved final genotype or phase accuracy. These fixed-input replays
are development controls, not an untouched accuracy estimate.


### Follow-up parameter and retention experiments — 24 September 2026

Including the deployed direction, contamination and reciprocal-family terms in
the training-fold predictive mixture did **not** solve the calibration problem.
Across the same 24 frozen-score cohorts, the private family-consistent fit gives
6,140/6,240 exact configurations, five false partial edges and 86 missing true
partial edges. Compare 6,194 / four / 29 with calibration disabled and
6,206 / three / 17 with pooled predictive calibration. It remains private.
No truth-derived generation constraints, thresholds or per-seed constants were
introduced to repair these outcomes.

The joint sample-effect/heterozygote-dispersion model estimates read parameters
well and improves untouched-window predictive scores. Two full-chromosome,
5x controlled read-stress comparisons nevertheless show long-path regressions:

| Input / observation arm | Called founder alleles | One-to-one wrong calls | Wrong + missing truth alleles |
| --- | ---: | ---: | ---: |
| Seed7004 chr10, shared binomial | 1,796,641 | 85,658 | 91,945 |
| Seed7004 chr10, joint | 1,799,465 | 89,122 | 92,585 |
| Seed7004 chr10, joint + candidate bank | 1,799,444 | 89,065 | 92,549 |
| Seed7019 chr7, shared binomial | 3,267,960 | 401,952 | 417,906 |
| Seed7019 chr7, joint | 3,271,697 | 423,703 | 435,920 |
| Seed7019 chr7, joint + candidate bank | 3,271,587 | 422,250 | 434,577 |

All these arms produce six chromosome-length rows in one component. The truth
panels contain 1,802,928 and 3,283,914 alleles respectively. These are founder
alleles, not painted sample alleles. Independent 200-marker rematching reduces
the shared/joint/bank wrong-call counts to 2,439/1,650/1,543 on chr10 and
42,690/36,812/37,464 on chr7. Such local rematching diagnoses path alignment,
but is not chromosome-long accuracy or proof of identifiable ancestral phase.

An ablation fitting sample means without dispersion gives 85,962 and 474,541
long-path errors respectively; the trade-off is not fixed simply by dropping
dispersion. Joint-model genotype posteriors using training-fitted mixture
weights improve held-out log loss, Brier loss and genotype error in both cases.
Those mixture priors are evaluation-only, never injected into inference GLs.
The joint model also improves held-out AD prediction in both real datasets;
see [read calibration](read_calibration.md). The joint-capable observation
selector and symmetric count-up/refit were promoted on 24 September with
explicit acceptance of the known accuracy/coverage trade-offs. The separate
candidate bank and family-consistent pedigree calibrator remain private.

A symmetric count-up/refit search using the unchanged founder objective was
tested on nine regional/full-chromosome controls. It recovers a fifth row on
one 128-block rare-founder region, reducing wrong-plus-missing truth alleles
51,200→28,137, but adds 1,623 wrong calls and still rejects the sixth row.
The other eight controls retain their founder counts; one full chromosome
changes by two fewer wrong calls. This is not a general retention solution.
These count-up controls used candidate-bank panels as fixed inputs; promotion
of the refiner does not also enable multi-model candidate-bank discovery.

The frozen original shared-read-calibration continuation for seed7010 finishes
21 paired chromosomes: common genotype errors 2,044,125→1,987,077 among
1,062,583,112 complete genotypes, and switches 3,586→3,379 on 168,787,880 common
phase comparisons. Reference chr3 exhausts its original phase-stability retries,
so this is **not** a completed 22-chromosome phase comparison. Its incomplete
output is not released and no whole-genome completion marker is fabricated.

### Exact Mendelian-exclusion acceleration

Exact per-sample likelihood-pattern encoding avoids recalculating the same
log-bet values at repeated markers. No likelihood rounding, window, missingness,
replacement parameter or evidence averaging changes. Samples exceeding the
bounded pattern catalog use the original direct scorer. Genetic-score caches
remain reusable; exclusion and decision identities record the new helper.

Twenty-six focused fixtures and four raw-chromosome kernel controls preserve
all evidence values exactly. Two full 22-chromosome public decision/release
replays preserve every evidence array and checked pedigree table bit-for-bit,
including successful chromosome-checkpoint resume.

Matched 36-CPU, four-chromosome-worker runs reduce complete exclusion time from
85.97 to 64.65 seconds for seed7009 (N320), and 52.40 to 33.18 seconds for
seed7019 (N160): approximately 25% and 37% less elapsed time. These include raw
evidence loading and checkpoint writes, but are **not whole-pipeline timings**.
Dynamic thread reassignment remains enabled. Encoding needs two additional
bytes per sample/marker; the worst-case pattern lookup is about 49 MiB per
numerical worker, usually much smaller at 5x.

## Length-independent founder refinement — 23 September 2026

Founder refinement now uses a sample-diplotype switch cost of 20 at every final
L1–L4 refinement level, instead of increasing the cost with component length.
The hierarchy's own length-scaled scorer, earlier local feedback, candidate
budgets, genotype-fit guards and missing-data rules are unchanged. This is an
approved scientific-model trade-off, not a numerically equivalent optimization
or an estimate of the biological recombination rate.

The controlled study froze discovery, balanced feedback and prepared local
panels, then replayed final assembly. It covered 89 distinct seed/chromosome
pairs, including two ordinary whole-genome controls and difficult crosses;
these are not 89 independent genomes. Truth was used for evaluation, never
per-chromosome model selection. Eight unchanged baseline replays reproduced
the saved called and missing arrays up to founder-row permutation.

| Evaluation set | Baseline founder errors | Refinement-only cost 20 |
| --- | ---: | ---: |
| 28 difficult chromosomes | 10,533,687 | 9,533,518 |
| 22 additional chr16 controls | 2,780,377 | 2,534,290 |
| Ordinary seed7000, N160, all 22 chromosomes | 17,920 | 15,751 |
| Ordinary seed7023, N320, all 22 chromosomes | 1,386 | 3 |

Errors are whole-component one-to-one founder allele mismatches, not errors in
painted children. Matched called alleles change from 156,077,306 to 156,038,588
on the difficult set and from 48,207,969 to 48,203,312 on the additional controls.
Errors plus missing alleles also improve: 10,699,123 to 9,737,672 and 2,807,188 to
2,565,758. Each ordinary genome has about 53.74 million matched founder calls;
N160 loses three calls and N320 calls are unchanged.

The new rule worsens 8/28 difficult chromosomes and 10/22 additional controls.
The largest difficult-set regression is seed7022 chr3, 529,520 to 762,795 errors;
seed7010 chr3 changes from 661,312 to 771,622. Much of the aggregate improvement
is chromosome-scale orientation: locally rematched errors plus missingness
improve only about 1.65% on the difficult set.

Both ordinary pedigrees remain exact (160/160 and 320/320), with no extra edges.
In read-stressed controls, seed7010 changes from 133/160 to 134/160 exact parent
sets but gains one false edge; seed7019 changes from 138/160 to 137/160 and from
two to three false edges. Those are real regressions, not floating-point noise.
These downstream checks used all 22 chromosomes but repainted only the changed
founder chromosomes; family phase and recombination were not rerun.

On 22 paired chr16 replays, with 56 CPUs per task and two tasks per 112-core
node, summed assembly time falls from 215.42 to 175.41 minutes (18.6%); the
median paired reduction is 8.1%. This is neither an end-to-end timing nor a
76-core benchmark. Changed realized search effort contributes.

Cross-fitted read-model calibration gave about 20% fewer founder errors on
three read-stressed chromosomes, but did not improve the paired downstream
pedigrees overall. Cheap before/after-refinement selection gave smaller and
mixed gains. Neither experiment is included in this default.

Assembly identity backend v20 records the refinement cost explicitly; refinement
diagnostics use model v13. Older scientific checkpoints must not silently resume
under the changed model. The assembly code identity can also invalidate feedback
caches despite unchanged feedback mathematics. Accepted results are preserved.
The frozen study and detailed results remain in
`.work/assembly_stagewise_20260923.NrgHtNdQ/`.

### Production promotion checks

The production workspace and standalone exchange fallback match the approved
prototype's fixed cost. Focused 40/6000-marker fixtures passed missing-allele
preservation, neutral observations, fallback/workspace parity, real forkserver
imports and checkpoint roundtrip/identity rejection. Syntax and diff checks pass.

Fresh production replays of seed7000 chr4 and chr5, from the same frozen feedback
and preparation, exactly match the prototype's called and missing arrays after
one-to-one row permutation: 4,305,348 founder-allele positions, including 27
missing values. New checkpoint resume is identical; actual old identities are
rejected. These checks took 204 and 155 seconds on separate 112-core nodes, with
resume taking under five seconds each. They confirm promotion parity, not an
additional independent accuracy gain. Logs and isolated checkpoints are in
`.work/refinement_default_20260923.6DCz7gAQ/`.

## Converged family decisions and full-marker release — 22 September 2026

The default retains four initial reciprocal passes for the frozen ancestry-path
factor, then converges the final family messages to maximum log-message change
`1e-8`. Try 128 undamped iterations, then restart the same equations with damping
0.5 for up to 4096 iterations if necessary. Residuals are measured before damping;
smaller steps cannot falsely signal convergence. Failure of both attempts is
reported, not silently published.

The checkpointed workflow also uses full-marker Mendelian exclusion and already
supported directions to resolve M0/M1 counts without adding edges. Bootstrap/LOCO
diagnostics remain separate from that fixed-genome release.

### Corrected worker-level validation

Promotion exposed an important limitation in the earlier private prototype:
its parent-process monkeypatch did not propagate into forkserver bootstrap
workers. Full-fit and LOCO solves used convergence, but parallel bootstrap fits
retained the bounded solver. The prior claim that every prototype bootstrap solve
converged is withdrawn. Production now runs the same solver in every worker.

Actual-worker validation covers 23 scored datasets, seeds7000–7022, each with
1,000 chromosome bootstraps and ordinary LOCO/prior checks:

| Dataset group | Exact Tier-B configurations | Supported edges |
| --- | ---: | --- |
| 21 non-read-stress datasets | 5,600/5,600 | All true edges, no extras |
| Seed7021, included above | 320/320 | 460/460 true, no extras |
| Seed7009, included above | 320/320 | 476/476 true, no extras |
| Read-stress seed7010 | 133/160 | 261/280 true, no extras |
| Read-stress seed7019 | 138/160 | 267/280 true, two pre-existing false reverse edges |

The total is 5,871/5,920 exact configurations and 8,315/8,347 true supported
edges, with two extras. Read-stress cases remain imperfect. Seed7022 was not
used to develop this fix. These are decision/control replays on frozen
upstream inputs, not 23 new discovery/assembly simulations.

A real seed7021 bootstrap fit stalled at residual 0.0102 after 128 undamped
passes; damping 0.5 reached 9.2e-9 in 207 passes. Damping 0.25, 0.5 and 0.75
independently reached the same fixed point on the captured case. One seed7010
resample required more than 512 damped passes, motivating the final 4096 ceiling.
Some already-completed controls used the lower ceiling: all their solves reached
the same tolerance before it. Raising an unused ceiling does not change any
iteration or result. No likelihood, pedigree prior or statistical release
threshold was changed to solve these numerical failures.

### Integration and interpretation

- Fresh pedigree preparation and genetic scoring from existing seed7021/7009
  paintings exercise the current optimized pipeline, followed by actual-worker
  inference, full-marker release, exports, supported-edge consumption and resume.
  Both recover the exact configurations and edges above.
- From saved genetic scores, decision plus new full-marker scans took about
  40–42 seconds on 112 CPUs, including 22–24 seconds for the scans. A subsequent
  decision replay reused all 22 exclusion checkpoints; direct final-result resume
  also passed. These are not complete pipeline runtimes.
- Four full-marker workers share the 112-core budget. Logs confirm expansion
  from 28 threads each through 37/38 and 56 to 112 for the final worker.
- The analytic dyad kernel matches the validated reference on uncertain/missing
  observations. Wholly missing and empty marker sets are neutral.
- Release fixtures cover caller eligibility, graph conflicts, ambiguous
  direction, incomplete candidate evidence and disagreement with full-data
  parent count. Disabling exclusion requires no full-marker input files.
- On all five saved full-marker controls, production release logic matches the
  prototype given the same input decisions. Downstream edge sets and original
  bootstrap/LOCO columns are unchanged *by the count-release step*. The corrected
  bootstrap solver can legitimately change those support values.
- Across the earlier full-marker scans of 7004, 7009, 7010, 7019 and 7021, no true
  dyad was falsely excluded among 2,096 evaluation controls. The same exclusion
  kernel/model is used in production. Truth was used only for evaluation.
- Package syntax, workflow defaults, output/decision identity coverage and
  `git diff --check` pass.

The exclusion bound assumes calibrated likelihoods and conditional read-error
independence. Directional release additionally assumes supported directions are
correct: it is **not an unconditional 99% pedigree guarantee**. See the
[method and output interpretation](pedigree_direction.md#fixed-observed-genome-m0m1-release).

This promotion leaves preparation/genetic-score cache identities unchanged.
Older campaign caches predate separate transmission-performance edits, so the
two integration runs correctly refreshed those stages. Accepted upstream run
roots and frozen campaign source were not modified. Artifacts, including failed
numerical attempts, are in `.work/pedigree_default_20260922.kqu6nMTI/`. The
preceding prototype report has an explicit worker-validation correction.

## Release cleanup and descriptive stages — 20 September 2026

The workflow cleanup shares downstream orchestration and reporting, scopes run
logs, removes unreachable pedigree scaffolding, and replaces historical stage
codes with descriptive names throughout the public source and CLI. It does
not change scientific models, priors or release thresholds.

Validation against the frozen pre-cleanup working tree included:

- Six saved 22-chromosome pedigree controls (820 samples) with identical
  Tier A, Tier B and complete relationship tables at the default 1,000
  chromosome bootstraps. These cover backcross, deeper-generation,
  missing-parent, N80 and N160 inputs.
- A fresh matched three-chromosome simulation (40 samples, 5×, seed920047)
  exercised discovery stop/resume, chromosome sharding, assembly, painting,
  pedigree, final phase, maps and evaluation. All 135 exported arrays matched
  exactly, including simulated reads, founder calls/missingness, paintings
  and final sample phase.
- The indexed AD-BCF fixture completed the same general reconstruction and
  resume route. All 132 exported founder/painting/final-phase arrays matched.
  Across both integrations, compared scientific CSV fields were unchanged;
  elapsed-time fields were excluded.
- Both short integrations released no pedigree edges: they validate
  orchestration and output parity, **not pedigree or recombination accuracy**.
  The saved full-genome replays above test the pedigree decisions separately.
- Eight observed-reference reporting fixtures retained the same numerical
  comparisons. Column names now distinguish observed-reference consistency
  from known founder truth or proven chimeras. Sample-selection policies
  and reference inclusion/withholding are unchanged.
- All 142 package modules and six CLI help routes imported successfully;
  scoped logging, sample/GL/map/CPU handoffs and shard boundaries passed
  focused checks. A wheel built without dependency installation and its CLI
  ran from the extracted artifact, with no private files or compiled caches.

The matched simulation emitted a non-fatal multiprocessing semaphore-tracker
warning at shutdown; it and its resume both exited successfully. This is not
claimed to be a warning-free run.

This validation does not add a new full-genome discovery/assembly simulation
or rerun the real datasets. Old checkpoint directories, serialized type names
and source identities are intentionally not migrated: start renamed workflows
in a fresh run root, retaining frozen source for historical results.
Readable AstCal/Tropheops deliverables remain unchanged. Private source snapshots,
fixtures and comparison records are in
`.work/publication_refactor_20260920_K1BWViis/`, outside the public package.

## Current acceptance coverage

| Portion | Completed evidence |
| --- | --- |
| Current fresh end-to-end pipeline | Seed3000, N320/5×, all 22 chromosomes from simulation through recombination and truth evaluation; progressive refinement enabled |
| Block discovery | All 1,647 seed400 chr1 blocks; repeated batches and 76 approximately 2× seed401 missing-data controls |
| Per-round balanced feedback | 9,287 local blocks in six seed/chromosome cases; full chr12 production parity plus bounded L1–L4/painting and resume checks |
| Pre-L1 founder completion | 5,484 chr16 blocks across seeds400–402; independent 2×/3×/5× crossfits |
| Dense q16 assembly and painting | All 22 seed402 chromosomes plus six controls; block discovery and painter held fixed |
| Final founder-path refinement | All 22 current balanced-input seed402 and seed403 chromosomes plus seed400–402 controls; geometry, missingness, provenance and resume checks |
| Guarded multiscale refinement | All 22 seed403 chromosomes plus six controls; fresh chr4 final assembly, painting and typed painting replay |
| Progressive final L1–L4 refinement | 98 N80/5× final-assembly comparisons across nine seeds, including 27 independent controls; two one-allele exceptions documented below |
| Earlier final-only optimized refiner | 217 N80 chromosomes across ten seeds; seven distinct N320 chromosomes; frozen-input, field-by-field and resume comparisons |
| N320 fragmentation replay | Eight previously fragmented chromosomes now continuous; seed407 chr15 final-refinement regression remains unresolved |
| Painting numerical implementation | All 22 seed402 q16 chromosomes, 148 components and 8,956,852 sites, plus seed400/401 chr16 controls |
| Packed painting evidence reuse | Full chr3/chr16 repaints, prepared fields and typed checkpoint round trips; memory-limited/uncached controls |
| Metadata-free pedigree | 20 full-genome finite-direction controls at N80/N160/N320, missing-parent and backcross controls; earlier seed401/402 wiring checks |
| Family phase correction | All 22 chromosomes of seeds400–402 on frozen earlier q4 inputs; later complete dense q16 seed403 run before multiscale refinement |
| Latest family phase dirty tiles | Two complete seed401 chromosomes; exact final phase/stability fields |
| Latest recombination indexed trials | All 22 seed401 final-phase maps; variable/zero maps, missingness, overlapping flips and reset controls; later complete seed403 run before multiscale refinement |

A fresh seed3000 run has exercised the current progressive implementation
from simulation through recombination maps and known-truth evaluation.
The earlier fresh seed403 run exercised simulation through recombination maps
with the balanced local feedback and final founder refinement. A subsequent
22-chromosome optimization comparison held its block discovery inputs fixed and reran
feedback, L1-L4, founder refinement, painting, pedigree inference, family phase
refinement and maps. Fixed-input comparisons must not be relabeled as fresh
simulations or fresh block discovery.

## Fresh seed3000 end-to-end validation

Completed 17 September 2026 with 320 samples (20 F1, 100 F2, 200 F3), 5× mean
depth, 2% read errors and 5 cM/Mb generating/inference defaults. Existing
empirical founder-sequence templates were inputs; the pedigree, reads, block
discovery, both balanced feedback rounds and downstream inference were fresh.
The run used dense bounded assembly and progressive refinement after each
executed final hierarchy level. No inference parameter was tuned against this
seed's truth, and production scientific code was unchanged during the run.

| Quantity | Result |
| --- | ---: |
| Chromosomes with one component and six long founder haplotypes | 22 / 22 |
| Founder markers represented | 8,956,852 / 8,956,852 |
| Called founder alleles | 53,740,022 / 53,741,112 |
| Uncalled founder alleles | 1,090 |
| Founder allele mismatches | 2,581 (48.03 per million called) |
| Exact metadata-free pedigree configurations | 320 / 320 |
| Correct M0 roots / M2 pairs | 20 / 300 |
| Correct edges / extra edges / missing edges | 600 / 0 / 0 |
| Called final sample alleles | 5,728,263,776 / 5,732,385,280 (99.9281%) |
| Final phase switches / eligible comparisons | 784 / 495,689,840 |
| Genotype errors / called genotypes | 486,200 / 2,864,125,767 |
| Component-aligned sample allele errors | 1,856,972 |

A one-to-one match to all six truth founders gives the same 2,581 founder
mismatches; the founder count was not forced. Errors are concentrated on chr3
(2,257), chr13 (166) and chr23 (115). Truth-to-panel errors plus missing alleles
total 3,671. A continuous chromosome product can still contain unknown allele
intervals; continuity is not a claim that every allele or long-range phase is
correct. Founder errors, sample allele errors and phase switches are distinct
metrics and must not be substituted for each other.

All 22 family phase outputs satisfied the canonical phase-stability stopping rule, but
full latent posterior convergence was not achieved. Switch comparisons do not
cross missing calls, unsupported components or incorrect intervening
heterozygotes, so the switch total must be read alongside coverage and genotype
errors. The release is a stable conditional phase product, not a converged
marginal posterior.

All 22 recombination maps completed, with shared-family orientation fitting
converged on its screened candidates. On correctly inferred edges within the
same observable exposure, expected crossovers total 25,061.08 versus 25,076
true crossovers (ratio 0.999405). This is a conditional, exposure-matched
comparison, not complete genome-wide crossover recovery.

All canonical completion, schema and sample-order checks passed. Frozen source
matches the production package. A separate chr8 family-solver replay matched
the distributed output's identity, every final allele call and its 25-iteration
stopping point. The [runtime report](performance.md#fresh-seed3000-n320-and-5)
distinguishes measured multi-node elapsed time from the estimated one-node
runtime and explains the run-local family phase scheduling.

This is one successful fresh seed, not validation of every cross design,
sample size or depth. It does not revalidate or resolve the earlier seed407
chr15 limitation below. The full readable report, canonical evaluation and
checkpoints remain in `work/runs/seed_3000/`; frozen source, founder metrics and
execution records remain in `.work/seed3000_full_20260917_cFrnkNJ8/`.

## Finite family direction

On 18 September 2026 the default pedigree direction policy changed from a hard
ancestry-layer gate to [finite continuous and family evidence](pedigree_direction.md).
All comparisons below use 22 chromosomes at 5x, no generation/parent metadata,
unchanged eligibility/exposure and release thresholds, and truth only for
evaluation. Assembly, painting and the genetic likelihood scorer are fixed.

| Saved full-genome controls | Seeds | Baseline exact Tier B | New exact Tier B |
| --- | ---: | ---: | ---: |
| N160, 20/60/80, seeds4000–4015 | 16 | 2,555 / 2,560 | 2,560 / 2,560 |
| N80, seeds2003/2005 (20/30/30), seed2010 (10/30/40) | 3 | 240 / 240 | 240 / 240 |
| N320, seed3000 | 1 | 320 / 320 | 320 / 320 |

The new model recovers all 3,120 individual configurations at **both** tiers,
including all 390 roots as M0, with no missing or extra edges. Full-data local
calls have no graph conflicts: cycle removal is not rescuing these roots.
The five N160 baseline failures were abstentions; their correct pairs already
ranked first genetically but failed direction or resampling support.

Missing-parent controls remove a seeded 25% or 50% of samples from each N160
input: 32 tests, 3,200 retained sample evaluations comprising 815 M0, 1,131 M1
and 1,254 M2 targets. Tier B improves 3,194 to 3,200 exact, without extra edges.
These controls initially hold the full-cohort background and screen fixed;
they are not 32 independent fresh simulations. Full-cohort permutations also
preserve the results. Three additional tests remove rows before recomputing
pedigree screening, background and likelihoods from saved painting inputs
(seed4000/75%, seed4006/50%, seed4011/50%). Both new tiers recover 280/280;
the old Tier B recovers 279/280. Upstream assembly/painting remains fixed.

Two focused N40 backcross simulations each contain eight generated roots,
16 ordinary offspring and 16 offspring crossed to one actual parent.
They use six supplied founder haplotypes, 5x reads with 2% errors and
5 cM/Mb, with one test adding contiguous missing reads and partial founder
sequence. Read-derived source factors, not inherited truth labels, enter pedigree.
Tier B improves from 24/40 and 26/40 to 40/40 in each. Tier A is 40/40 and
33/40; its remaining calls abstain or leave identities unresolved, with no
extra edges. This checks painting-source/pedigree backcross inference, **not** discovery
or assembly of an entire backcross dataset.

An initial single-pass family correction lost root support under sample
withholding and was rejected. Four synchronous cavity-message passes resolved
that issue. Four versus eight passes produced identical relationship tables
on six controls; serial versus 112-worker execution preserved tables and
inspected support columns on seed4011. Small tree factors match exact
enumeration. A log-odds implementation also handles strong reverse messages
without rounding away a small probability before removing that message.
After this correction, all 20 full-seed tables and all missing-parent/backcross
metrics above remained unchanged.

This is empirical support for the tested designs, not proof of calibrated
parenthood probabilities or general correctness. Loopy family dependence,
painting errors, incompletely screened alternatives and structured missingness
remain limitations. Family phase/recombination were not rerun after this pedigree-only change; the
fresh end-to-end results elsewhere on this page refer to their recorded
earlier code versions. Detailed attempt logs and isolated outputs are retained
locally in `.work/pedigree_direction_20260918_UtLS28/`.

## Joint short-ancestry paths

On 20 September 2026 the family-direction default gained a one-way
[joint ancestry-path correction](pedigree_direction.md#joint-short-ancestry-paths).
This is a scientific decision-model change, not a new genotype likelihood or
an assembly modification. Release thresholds, fixed state priors, candidate
eligibility and the top-20 M2 panel are unchanged.

In the F4–F6-only seed5001 simulation, F4_9 was correctly selected as M0 in the
full fit but withheld at Tier B: M0 bootstrap support was 553/1,000, against a
strong alternative assigning its double grandchild as parent. The new factor
uses uncertain observed two-edge ancestry paths to penalize that reversal,
without using generations, sample names or simulation truth. Both focal
parents are considered jointly so legitimate backcrosses are not penalized by
a redundant reciprocal constraint.

| Seed5001 quantity | Reciprocal-only default | Joint-path default |
| --- | ---: | ---: |
| Exact Tier-B configurations | 299/300 | 300/300 |
| Exact Tier-A configurations | 297/300 | 300/300 |
| Target M0 bootstrap support | 553/1,000 | 1,000/1,000 |
| Target M0 leave-one-chromosome-out support | 18/22 | 22/22 |
| Correct edges / extra edges | 400/0 | 400/0 |

Prototype validation covered 31 full-cohort/rescored controls: seeds4000–4015,
N80 seeds2003/2005/2010, N320 seed3000, deep-generation seed5001, complex-cross
seeds5000/5003/5004/5005/5007, two supplied-founder backcross controls and three
rescored missing-parent subsets. All 4,580 configurations are correct at both
tiers. These include random missing individuals, 50%/100% backcrossing, and
combined backcross/missing-individual designs. The structured-missingness
supplied-founder control additionally improves Tier A from 33/40 to 40/40.

Another 32 decision-only sample-withholding tests preserve 3,200/3,200 Tier-B
configurations. Their three Tier-A abstentions were already present before
this change. Six sample-order permutations and configuration budgets 8/16/32
on four controls preserve relationship tables and state-bootstrap support.
These are repeated evaluations of cached datasets, not independent fresh
end-to-end runs.

The integrated canonical code, including normal forkserver bootstrap workers,
reproduced all 31 full-cohort/rescored controls and three representative
withholding controls, matching prototype tables and inspected resampling
support. Full and partial caller-chronology tests each recover 40/40 at both
tiers. Twelve exact-enumeration checks cover joint configurations, shared
intermediates, co-parent conditioning, chronology exemptions and conservative
omitted-mass bounds; six full-score fixtures match the accepted prototype.
Checkpoint round trips retain the Tier-B schema, and pedigree identities include
the new configuration and source module.

An initial edgewise variant lost two Tier-B backcross calls and was rejected;
the accepted joint factor avoids that redundant conditioning. The factor
remains a composite/loopy approximation, not a calibrated global pedigree
posterior. The independent deep-generation seed5006 is still running and is
not counted as validated here. Assembly, painting, family phase refinement and
recombination maps were not rerun for this integration.

Detailed experiment artifacts are retained locally in
`.work/pedigree_ancestral_direction_20260920_fIUxZk1F/`; canonical integration
checks are in `.work/pedigree_path_default_20260920_YdXINPmE/`.

## Progressive final L1–L4 refinement at N80

The complete refiner now runs after each executed level of **final** L1–L4
assembly. Neither of the two local feedback/context passes runs it.
The controlled comparison reused identical cached, post-feedback local panels
and original genotype likelihoods in both arms: the frozen predecessor
`b6bc14e` with final-only refinement versus the progressive implementation.
These are final-assembly replays, not new simulations or fresh discovery runs.

All 98 comparisons completed at N80 and 5×. Four complete 22-chromosome
genomes use the 20/30/30 design:

| Seed | Called-allele errors, final-only → progressive | Called alleles, final-only → progressive | Missing alleles, final-only → progressive |
| --- | ---: | ---: | ---: |
| 2002 | 28,609 → 28,565 | 53,712,295 → 53,712,290 | 28,817 → 28,822 |
| 2003 | 107,211 → 104,706 | 53,701,807 → 53,701,743 | 39,305 → 39,369 |
| 2005 | 1,985 → 1,834 | 53,725,799 → 53,725,794 | 15,313 → 15,318 |
| 2006 | 10,746 → 8,949 | 53,713,569 → 53,713,607 | 27,543 → 27,505 |

Together these improve 148,551 → 144,054 errors among approximately
214.85 million called founder alleles. Calls decrease by 36 overall, so
errors plus missing alleles improve by 4,461 rather than 4,497.
This is not evidence that every remaining error is recoverable.

The other ten comparisons are seed2000 chr13, seed2001 chr20, five seed2004
chromosomes, seed2010 chr18 and seed2011 chr8/20. The last three use the weaker
10/30/40 design. Seed2010 chr18 improves 100,865 → 52,130 errors; seed2011 chr20
improves 17,164 → 4,869. These deliberately difficult controls dominate the
pooled improvement and should not be treated as a representative error rate.

All 22 seed2003 chromosomes and seed2004 chr3/4/13/20/23 were declared as
independent validation before observing their outcomes. They were not used to
develop this change, although they are existing project simulations rather
than globally untouched or freshly generated data. Those 27 cases improve
107,772 → 105,267 errors, with 64 fewer called alleles. No inference parameters
were retuned against their truth.

Across all 98 cases, 21 improve their called-allele error count, 75 are unchanged,
and two have one extra error each. Every progressive output is one component
covering all input SNPs, with six rows matching six distinct truth founders;
one-to-one matching gives the same forward error count. The count was not
forced. There are two explicit exceptions to strict no-regression equivalence:

- Seed2006 chr16: 76 → 77 errors, with 37 additional calls. Of those new calls,
  36 are correct and one is wrong. Among previously called alleles, two errors
  are corrected and two introduced. Errors plus missing alleles improve by 36.
- Seed2003 chr16: 4,698 → 4,699 errors with unchanged coverage. One original
  local-row choice changes one allele at approximately 5.93 Mb; both the
  canonical primary and full-site secondary scores tie exactly. No rule was
  added to choose this allele using truth. This is the only case where either
  forward errors-plus-missing or reverse truth-to-panel errors/missing worsens,
  by one.

The implementation is retained for its overall accuracy/coverage improvement
and bounded runtime cost, **not** as a claim of perfect per-site preservation.
The original naive progressive versions had larger regressions; these were
traced to partial-founder evidence ignored at primary-score ties and to useful
primary-neutral alternatives not returned by the unrestricted search. The
[secondary tie rule and primary-preserving search](methods.md#partial-evidence-at-primary-ties)
address those mechanisms without changing the primary mask, inventing alleles,
or selecting against truth. Early proposal coarsening remains a documented
search approximation.

Focused checks covered 24 real panels against the existing single-site
predictive scorer (identical scores), complete-input equivalence, neutral
unobserved samples, restricted-path primary invariance, serial/parallel
component results, completed resumes for all 98 cases, partial component reuse,
and exclusion from both feedback levels and the explicit off setting.
Partial-resume soft probabilities differed by at most 2.98e-8 due to the
existing fresh-hierarchy float32 conversion; calls and provenance were exact,
and probabilities were identical at float32 precision. All paired local-input
identities and local truth metrics matched.

This validates final assembly, not a fresh painting–recombination rerun. The earlier N320
seed407 chr15 limitation below has **not** been revalidated or established as
fixed. Timing scope and CPU budgets are recorded in
[performance](performance.md#progressive-final-assembly-at-n80).
Frozen code, per-level checkpoints, per-chromosome CSV/JSON metrics and
diagnostics remain in `.work/progressive_refinement_20260917_V2eVeIBe/`.

## Optimized founder refinement

The earlier final-only performance comparison starts from unchanged cached L4 inputs;
it does not repeat discovery, feedback or hierarchy. The 217 completed N80/5x
chromosomes cover seeds2000–2006,2010,2011 and 19 available chromosomes from
seed2012, spanning both 20/30/30 and 10/30/40 cohort designs. Every final product
matches its independently recomputed frozen pre-optimization baseline:
component geometry, ordered called/missing alleles, inference arrays,
probabilities, atomic provenance and phase-boundary metadata. Complete resumes
match for all 217, as do targeted partial resumes. Combined founder errors are
unchanged at 1,164,185 / 530,043,572 called cells.

The heuristic deep-count screen avoided 163 losing refits across 529 components
and retained all 11 accepted count reductions. The largest cheap deficit of
an accepted reduction was 4.54 per-founder complexity costs, below the default
threshold of eight. This is empirical evidence, not a safe mathematical bound
or a calibrated probability. The unscreened setting remains available.

Seven distinct cached N320/5x chromosomes also match the frozen predecessor:
seed404 chr1/11/20, seed405 chr20, seed406 chr3, seed407 chr10 and seed408 chr19.
Paired warm repeats and one extra CPU-budget check bring this to 27 solves and
14 optimized-versus-predecessor field comparisons. Both implementations give
274 errors / 20,334,788 called alleles, 592 missing output cells and 470 reverse
truth-to-panel errors/missing. Every completed resume passes.

These checks establish preservation of the immediate pre-optimization output,
not universal correctness of the accumulated search changes. In particular,
the earlier cubic search redesign changed seed2006 chr20 from 1,927 to 2,785
errors among the same 1,769,283 called cells. Later exact comparisons use that
post-redesign baseline; they do not erase the earlier regression. Normalized
beam variants and an algebraically simplified short dual were rejected after
larger chromosome-level regressions and are not in the package.

Truth was used only for evaluation after inference. The N320 cached-L4 tests
preserve old fragment boundaries by design and therefore do not evaluate the
upstream fragmentation fix. That separate replay follows below. Timings,
cache conditions and memory measurements are in [performance](performance.md#final-founder-refinement).

## N320 fragmentation replay

An inventory of five stored N320/5x seeds found 102/110 chromosomes already
represented by one six-row component. The other eight were rerun from cached
block discovery blocks and original genotype likelihoods through both balanced feedback
rounds, current partial-founder-aware L1–L4, final refinement and unchanged
painting. No old feedback/hierarchy output was reused.

All eight now have one component with six rows spanning every input SNP.
Each reconstructed row matches a different truth founder across the whole
chromosome; a one-to-one assignment gives the same error count. The count
was not forced and truth did not enter inference. Second-round entirely unknown
local rows fall from 14 to zero. Final missing output cells fall from 3,112
to 259, with 23,100,869 called cells in the new products.

| Seed / chromosome | Components before → after | Founder-allele errors before → after | New called cells | New missing cells |
| --- | ---: | ---: | ---: | ---: |
| 404 chr20 | 3 → 1 | 245 → 18 | 1,769,856 | 0 |
| 405 chr4 | 6 → 1 | 5 → 2 | 2,276,784 | 0 |
| 405 chr17 | 5 → 1 | 44 → 44 | 2,043,402 | 60 |
| 406 chr3 | 3 → 1 | 28 → 8 | 9,034,891 | 191 |
| 406 chr8 | 5 → 1 | 19 → 23 | 1,537,476 | 6 |
| 406 chr11 | 5 → 1 | 30 → 18 | 2,345,106 | 0 |
| 407 chr15 | 3 → 1 | 2,075 → 4,994 | 2,123,634 | 0 |
| 408 chr18 | 5 → 1 | 17 → 51 | 1,969,720 | 2 |

Continuity passes, but this is **not an overall accuracy improvement**:
whole-component errors increase 2,463 → 5,158, chiefly due to seed407 chr15.
The other seven together improve 388 → 164. Old fragments match truth
independently, whereas joined outputs match across a whole chromosome;
splitting both products onto identical spans still gives 2,463 → 5,146, so
the overall regression is genuine. Seed408 chr18 also worsens on identical
spans; seed406 chr8 remains at 19 errors on those spans.

The largest regression localizes to final refinement of seed407 chr15:

| Stage | Old errors | New errors | Old / new errors on identical spans |
| --- | ---: | ---: | ---: |
| L1 | 250 | 320 | 250 / 269 |
| L2 | 83 | 153 | 83 / 102 |
| L3 | 1,756 | 1,826 | 1,756 / 1,775 |
| L4 | 1,756 | 1,826 | 1,756 / 1,775 |
| Final founder refinement | 2,075 | 4,994 | 2,075 / 4,994 |

All 2,919 excess final chr15 errors occur at approximately 37–40 Mb. A later
truth-only ancestry audit (20 September 2026) separates this headline result:

| Historical final product | All called errors | Errors where ancestry is represented | Errors where ancestry is absent |
| --- | ---: | ---: | ---: |
| Original | 2,075 | 16 | 2,059 |
| Fragmentation replay | 4,994 | 34 | 4,960 |

Thus 2,901 of the 2,919 additional errors (99.38%) are in founder ancestry absent
from every sampled individual. The replay has 2,110,984 represented called
alleles and 12,650 absent-ancestry called alleles. All 34 represented errors
have at least two carriers, which does not establish independent-lineage
support. This is an audit of the historical saved products, not a fresh run of
the latest progressive refiner, and it does not make unsupported calls correct.
No model was tuned against these labels. The 18-error represented increase
remains, but the large hotspot is predominantly outside sampled ancestry.
Neutral arrays, matching, and ancestry strata are retained in
`.work/release_tools_20260920_k0scCHon/chr15_historical/`.

Its final
local 200-marker errors are only 26 → 44, consistent with a predominantly
long-range path problem. This locates the observed deterioration but does not
prove its exact causal move, establish whether the affected ancestry is
identifiable from the samples, or isolate a particular optimization. It remains
an unresolved scientific limitation; joining more components is not itself
evidence of better long-range phase.

All eight typed painting/sample-order checks and complete checkpoint resumes pass.
Original results remain untouched, with these replay products isolated pending
scientific review. This is not a new 110-chromosome whole-pipeline validation:
only eight chromosomes were reassembled, block discovery was reused, and pedigree–recombination were
not rerun. Detailed artifacts are retained locally in
`.work/n320_refiner_20260917_g3ece5Py/` and
`.work/n320_fragment_retry_20260917_UTISglBg/`.

## Seed403 end-to-end performance comparison

Both runs used 76-core Ice Lake allocations, 320 samples and 5× mean read depth.
The optimization comparison preserved final founder component geometry and
every ordered called/missing founder allele on all 22 chromosomes. Pedigree
and per-chromosome integer evaluation metrics matched exactly; map summary
floating values matched within absolute 1e-8 plus relative 2e-9 tolerance.

- Metadata-free pedigree: 320/320 exact configurations, including all 20 roots,
  300 parental pairs and 600 edges, with no extra or missing edges.
- Final sample phase: 762 switches across 495,613,063 eligible comparisons;
  99.9202456% called allele coverage.
- Founder reconstruction: immediately after L4, 4,483 errors in 53,740,427
  called alleles; after founder refinement, 1,476 in 53,740,334 calls. These
  use closest-truth matching over each whole component, not sample phase
  errors or marker-wise matching. Chr4 accounts for 1,463 remaining errors in
  this performance baseline. The later multiscale refinement comparison below
  reduces the founder error total to 19; the pedigree and final sample-phase
  figures above have not been rerun with those changed founder chromosomes.

| Disjoint pipeline portion | Before (minutes) | Optimized (minutes) |
| --- | ---: | ---: |
| Feedback round 1 | 53.96 | 46.98 |
| Feedback round 2 | 61.35 | 55.40 |
| Final L1-L4 and evidence preparation | 52.46 | 46.57 |
| Final founder refinement | 19.56 | 9.45 |
| Painting computation | 1.38 | 1.41 |
| Pedigree evidence preparation | 4.07 | 3.81 |
| Global pedigree inference, including loading | 16.44 | 16.60 |
| Family refinement and phase correction | 16.71 | 16.10 |
| Recombination maps | 2.98 | 2.61 |

The measured run reusing block discovery took 204.03 minutes, excluding final evaluation.
Adding the unchanged baseline input/discovery durations gives a **252.53-minute
fresh-run estimate**, versus 282.87 measured baseline minutes (10.7% less).
This is not a measured fresh-seed optimized runtime, a 112-core measurement,
or evidence that every stage improved. Node differences, filesystem throughput
and native-cache warmth affect elapsed times. The single-read block discovery I/O change
was not timed by this cached block-discovery comparison.

A further coordinator optimization batches missing-data scans, fuses validated
snapshot copies, packs worker inputs in parallel and avoids large eligibility
gathers. It leaves thresholds, likelihoods, search order, missingness semantics,
checkpoint formats and the dynamic worker allocator unchanged.

The additional fixed-input tests matched all 134,382 focal-mask blocks across
66 input sets and all 88 hierarchy-level eligibility/boundary cases on the 22
seed403 chromosomes. Paired full chr4 preprocessing and its actual L1 output
matched exactly, including calls, probabilities and provenance. On warm,
reverse-order repetitions, preprocessing fell from 25.23 to 8.75 seconds
(65.3% less wall time); this is a component timing, not a whole-pipeline
speedup. It is not included in the preceding fresh-run estimate.

A 512 MiB shared-memory first-touch copy took median 0.273/0.105/0.058/0.087
seconds with 1/4/16/76 threads. The copy path therefore caps its explicit budget
at 16; compute kernels retain the full requested budget. Small checks cover 27
focal-mask, 12 eligibility, 9 snapshot, 6 packing and 22 transport cases, plus a
complete two-block preprocessing comparison against frozen reference code.
After integration and the copying cap, the production-import chr4 check passed
again: warm preprocessing 25.456 -> 9.065 seconds (64.4% less), with identical
preprocessing and L1 outputs. The final helpers were not followed by another
full-genome reconstruction; their acceptance uses the exact all-chromosome
mask/boundary comparisons and the repeated full chr4 integration.

## Founder reconstruction and assembly search

L1–L4 share one distance-aware linker with a maximum of 20 fitting iterations.
Dense transitions and 16 full proposal scores per category are the default.
Structured transitions are an optional restricted model, not a numerically
equivalent dense implementation.

The initial four-score budget caused major localized reconstruction losses and
was rejected. On all 22 seed402 chromosomes, q16 reduced closest-founder
mismatches from 50,866 to 6,243 and increased painting coverage from 98.9716%
to 99.9133%. On the same 493,468,946 eligible phase comparisons, switches were
2,128 for q4 and 2,130 for q16: broader search did not improve every metric.

Across chr16 of seeds400–402, q16 produced 1,073 founder mismatches versus 854
for broader reference search, over 6,579,146 called founder cells. These are
allele mismatches, not sample phase-switch errors. A selected q32 case improves
reconstruction, but a full-genome q32 pedigree has not been validated.
See [assembly model comparisons](founder_scaling.md) for denominators and trade-offs.

The earlier model-preserving assembly optimizations retained every compared
L1–L4 digest, founder truth metric and unchanged-painter output in their
controlled chromosome checks. The subsequent non-assembly update did not edit
L1–L4 algorithms, transitions or q16 budgets. The final refinement below is a
subsequent scientific search/objective change, not an equivalence-only speedup.

## Final chromosome refinement

On current balanced seed401 chr10 inputs, local error count is 57, but errors
rise to 218 at L1, 3,065 at L2 and 11,925 at L3/L4. The no-feedback chromosome result
has 7,600 errors. Traces show loss of locally supported row choices in selected
intermediate panels, plus search failures to propose better same-count paths.
Reopening the original prepared rows addresses that hierarchy bottleneck.

| Final chromosome founder allele metric | Before refinement | Default refinement |
| --- | ---: | ---: |
| Seed401 chr10 errors | 11,925 | 1,300 |
| Seed401 chr10 called alleles | 1,802,726 | 1,802,676 |
| Seed401 chr10 truth-to-panel errors/missing | 11,927 | 1,352 |
| All 22 seed402 chromosome errors | 2,181 | 54 |
| All 22 seed402 called alleles | 53,741,081 | 53,741,079 |
| All 22 seed402 truth-to-panel errors/missing | 2,212 | 87 |

All 22 seed402 chromosomes improve or tie, with just two additional missing
output cells; component spans/counts and founder counts are unchanged. Additional
controls improve: seed400 chr10 6→1, chr16 390→254; seed401 chr13 35→0,
chr16 75→2. These are development controls, not a new held-out biological cohort.
The main chr10 retains three components; the refiner does not bridge its gap.

Each output founder is compared with its closest true founder over the entire
component, without a one-to-one constraint. The reverse truth-to-panel metric
also penalizes missing calls and detects lost variation. These are founder SNP
allele errors, not sample phase-switch counts. In particular, the main chr10's
remaining errors are more fragmented: its diagnostic supported local founder-label
changes rise18→66 despite the much smaller total number of wrong alleles.
That diagnostic is not a sample crossover count.

The implementation checks full-site scores against a reference, exact unordered
states and linear traceback against the original beam, missing and flat evidence,
composed provenance, metadata commutation, input preservation and partial/complete
checkpoint resume. A36-block canonical run exercises both feedback rounds, final
refinement, unchanged painting and typed painting replay; refinement is absent from
the L1/L2 context checkpoints. A fresh full chr10 final hierarchy/refiner run
reproduces the result and passes unchanged painting, typed painting publication and
exact assembly replay. CLI/TOML/environment configuration is checked for all three
workflows, including reuse of local feedback checkpoints when final refinement
is toggled off. The clean implementation matches all 26 paired prototype outputs
exactly; seed401 chr16 is an additional clean-only control.

An actual fragmented seed406 chr3 resume exposed a boundary-annotation issue:
cached prepared leaves predate phase breaks introduced by the hierarchy.
Rebuilding a changed final component now preserves its unchanged outer break
flags, reasons and joint-information counts from the input component. A focused
reproducer retains identical allele calls and source paths while fixing the
annotation loss; the actual three-component chr3 replay matches the canonical
inference arrays and boundary metadata and passes completed resume. Components
were never joined by this issue. This is a metadata/resume fix, not an allele
accuracy gain or a change to the fitting objective.

The main refinement takes 435s on12 cores and 253s on38 cores of the measured Ice Lake
node, with identical scientific output. Large chr3 takes 235s on13 cores and 164s
on37 cores. These are additional refinement timings, not complete pipeline runtimes
or measured 112-core figures. Widening the maximum beam to 4096 gives exactly the
same chr10 output and likelihood; the default therefore retains the 1024 ceiling.

Weaker founder penalties, sample-guided proposals, extra local windows and several
alternative beam guides did not provide a better validated replacement. Some
increased read likelihood while worsening true allele accuracy. The accepted
strategy therefore relies on independent chromosome comparisons, not merely
monotone read scores. See the [model's limitations](methods.md#final-founder-path-refinement).

## Multiscale founder-path escape

Seed403 chr4 isolates another search barrier. Prepared local panels have 22
called errors, L1 has 9, L2 has 1,104, and L3/L4 have 3,264. The largest loss starts
in L2 around 28.82–31.63 Mb and expands in L3; L4 itself adds no further error.
Earlier final refinement stops at 1,463 errors on one path despite a better
whole-chromosome path being representable by the original local candidates.
An evaluation-only truth oracle confirms this is not simply missing alleles
or an objective that necessarily prefers the incorrect path.

L1-sized proposal moves escape that barrier, followed by ordinary fine-scale
refinement. They are chosen from read likelihoods without truth or pedigree.

| Founder allele metric | Earlier refinement | Guarded multiscale refinement |
| --- | ---: | ---: |
| Seed403 chr4 called errors | 1,463 | 6 |
| Seed403 chr4 called alleles | 2,276,611 | 2,276,611 |
| Seed403 chr4 truth-to-panel errors/missing | 1,636 | 179 |
| All 22 seed403 called errors | 1,476 | 19 |
| All 22 seed403 called alleles | 53,740,334 | 53,740,334 |
| All 22 seed403 truth-to-panel errors/missing | 1,655 | 198 |

Chr4's called/missing mask is identical, not just its call count. The six
remaining errors occur in five original local panels; each lacks an error-free
candidate for the affected true founder, and their minimum available error
counts sum to six. They cannot all be eliminated by choosing different whole
local rows. The final rate is 2.64 errors per million called chr4 alleles.

An unrestricted macro search was rejected: it worsened seed401 chr10 from 1,300
to 3,396 errors despite raising the penalized score. Its first larger move
saved four internal sample switches but worsened genotype fit by 0.861 score
units. The new genotype-fit guard rejects that move. Chr4's accepted larger
move instead gains5,452.534 genotype-fit units and 11,513.237 total score units.
This is evidence for the restriction, not proof that all accepted moves are
biologically correct.

All 28 guarded frozen-input controls pass: the 22 seed403 chromosomes and
seed400 chr10/16, seed401 chr10/16, and seed402 chr4/13. The other27 cases retain
their earlier error counts, call counts, founder counts and component geometry;
seed401 chr10 remains at 1,300. These are development controls, not untouched
held-out genomes. Checks cover exact macro emission packing, bidirectional
score agreement, genotype-score decomposition, missing/neutral evidence,
composed provenance and full/partial checkpoint replay.

A fresh chr4 final L1–L4 run reproduces every upstream founder allele exactly,
then reaches the same six-error result. Unchanged painting, typed painting publication
and replay pass; the installed production code also replays all six cached
assembly/refinement phases with identical founder alleles. No block discovery or feedback
regeneration was needed. pedigree–recombination were not rerun after this founder change.

On the 76-core node, fresh final assembly plus refinement takes 343.3s, of which
229.1s is refinement; painting takes 15.0s. The stored earlier 76-core chr4 refiner
measurement is 136.9s, so this case costs about 92s more. These are separate
stored-run measurements, not a new full-seed or 112-core benchmark. The full
refiner control at 38 cores takes 287.7s with the same six-error output.

### Default dual-decomposition escape pass

A second, all-founder search pass now follows the existing final refiner.
It retains the same local candidates, full-site objective, fixed founder count,
component spans, missingness rules and macro genotype-fit guard. It changes
proposal search, not the observation model; truth and pedigree do not enter
fitting. See [the method and cost](methods.md#final-founder-path-refinement).

Matched seed409 chr3 ablations distinguish its mechanism: another beam pass,
fixed-painting refinement alone and small-block-only dual search each leave
2,742 founder errors. Dual search over the same L1 pieces, followed by ordinary
polishing, reaches 81 across the same 9,034,784 called alleles. The first accepted
macro move improves genotype fit and removes one internal sample switch.
This is a search escape, not a changed scoring rule or a masking gain.

Whole-genome prototype confirmation covers 132 chromosomes across six N=320,
5x seeds (404–409), including a variable-map simulation. Two chromosomes improve
and 130 remain unchanged, with no
called-denominator or retained-variation regression. Across 322,435,082 identical
called founder alleles, errors decrease 54,598 to 51,563. The changed cases are
seed409 chr3 (2,742 to 81) and chr19 (376 to 2). The chr19 improvement was already
attainable by an earlier broader search; chr3 is the decisive new mechanism.
These are controlled additional passes after the completed original refiner,
not validation of replacing its initial searches from unrefined L4.

The full 22-chromosome downstream confirmation preserves 320/320 exact pedigree
configurations (20 zero-parent roots, 300 correct pairs and 600 edges, no extras).
Final genotype errors decrease 485,187 to 466,475; phase switches 692 to 677; called
alleles increase 5,725,594,073 to 5,726,754,475. Other 20 chromosome metrics are
unchanged. Matched-callable comparisons on both changed chromosomes confirm
genuine corrections rather than improvements solely from lost calls.
This is simulation evidence, not real-data trio ground truth or a global
optimality guarantee.

Production integration then passed four cached-input replays through the shared
assembly API and typed sample painting release. Both refinement passes recomputed
from frozen, unrefined L1–L4 results. The first pass matched the original output;
the final scientific founder and painting arrays matched the validated prototype
exactly, including component boundaries and missing-data fields.

| Production replay, all N=320 and 5x | Founder errors | Called founder alleles |
| --- | ---: | ---: |
| Seed409 chr3 | 81 | 9,034,784 |
| Seed409 chr19 | 2 | 1,636,206 |
| Seed406 chr3, fragmented control | 28 | 9,034,691 |
| Seed404 chr1, clean control | 0 | 1,976,345 |

Completed resume passed on all four. The clean control also passed interrupted
proposal resume and rejection of a checkpoint with a changed sweep setting.
Default-on/off wiring and 36 independent small numerical cases passed. These
checks reused cached upstream results; they are not another fresh whole-genome
or downstream run. The already validated downstream scientific inputs were
reproduced. The additional pass has a material runtime cost: 10.1–16.9 minutes
on the three shorter production controls and 51.0 minutes on the difficult chr3,
using full-node budgets of 76–128 CPUs. These are component measurements, not
a new whole-pipeline timing or evidence of efficient full-node scaling.

## Missing-data discovery and completion

Focused cases cover missing read cells, wholly unobserved samples, long missing
tracts, duplicate/rare founders, partial founder calls, wildcard assignments,
dtype differences, ties and multiple thread counts. Unknown alleles are not
treated as reference or as independently guessed copies of one shared founder.

All compared block discovery scientific outputs matched exactly. Across the 5,484
completion blocks, every discrete field matched and maximum float-array
differences were 1.01e-12. Completion tests separately checked positive dosage
marginals, exact shared-site cavity likelihoods, whole-bin exclusion, crossfits,
early stopping and deterministic transmitted-allele projection.

One exchangeable-founder synthetic case amplified roundoff in internal
diplotype probabilities to 7.96e-6, release confidence to 4.95e-7 and effective
carrier counts to 1.22e-5. Best starts, iteration counts, released calls and
held-out profiles were unchanged. This is not asserted to be bit-identical or
evidence that every floating difference is harmless.

The exact unresolved-founder cap remains six. Polynomial likelihood evaluation
and marginal folding do not remove the exponential output size of explicitly
enumerated configurations or approximate posterior mass.

## Local feedback selection

The accepted default selects after **each** context round. The comparison held
the original discovery inputs and local scoring rules fixed, with truth used
only for evaluation. It covered seed402 chr10/12/16/19 and seed401 chr10/13:
9,287 blocks in total.

| Local 200-SNP metric | Selection only at the end | Selection after each round |
| --- | ---: | ---: |
| Incorrect called founder alleles | 183 | 135 |
| Called founder alleles compared | 10,266,082 | 10,265,983 |
| Errors per million called SNP alleles | 17.83 | 13.15 |
| Truth-to-panel errors or missing alleles | 652 | 566 |
| Exactly represented distinct true local haplotypes | 51,275 / 51,503 | 51,291 / 51,503 |

Called-allele errors compare each output row to its closest true local haplotype
over called sites, without a one-to-one row constraint. Unknown cells are not
counted as called errors. The reverse metric compares each true row to its
closest output row and penalizes both wrong and unknown alleles; its denominator
is 11,140,146 true founder allele cells. Neither metric assesses cross-block
phase, final L4 chromosome accuracy or sample painting switches. The aggregate
improvement is not uniform: small local regressions and occasional extra rows
remain. These are development simulations, not a real-data accuracy estimate.

The full seed402 chr12 production feedback route reproduced all 1,794 selected
panels exactly in each round: 17 called-allele errors over 1,998,544 calls and
zero reverse errors/missing. A 36-block integration exercised both selections,
final L1–L4 and painting, mode separation, completed/batch/interrupted resumes,
and preservation of original discovery inputs. Missing-read, noncontiguous
marker-mask and genetic-map wiring checks accompany those comparisons.

## Partial-founder linking and empty feedback rows

The following N80 results describe the earlier partial-founder integration.
Later search changes and their regressions are distinguished in
[optimized refinement](#optimized-founder-refinement); the subsequent N320
replay is reported [above](#n320-fragmentation-replay).

Focused checks use seed2006, N=80 (20/30/30), 5x chr13 and small explicit
likelihood controls. The actual isolated 200-SNP block at positions
22,371,775–22,411,082 previously retained six rows, one entirely unknown.
Balanced post-feedback refitting removes that row with a penalized-score gain
of 2.6492902370. The five rows contain 200/200/199/200/200 called alleles.
The canonical two-round feedback path reproduces this result. It withdraws
one formerly called allele; it does not invent a value for the empty row.

Strict selection correctly vetoes this particular reduction because that
one backbone call would be lost. Both its calls and assignments remain stable
on repetition. A genuinely distinct singleton row in a small read-likelihood
control also vetoes deletion when deleting it worsens the fit. A partially
called rare row does not initiate deletion.

The seven compact predictive emission categories match an independent dense
genotype-distribution calculation. Complete-input linker weights are identical;
partial binned scores match the direct calculation within 1e-12. Forward and
backward compact/dense scans and the log-domain fallback agree within 1e-10.
Shared atomic unknown alleles retain the homozygous distribution rather than
acquiring spurious heterozygote mass. Dense and structured transition fitting
both consume partial inputs. Wholly unsupported blocks and explicit phase
breaks remain unresolved.

The seed2006 chr13 end-to-end assembly/painting check now produces one
six-row chromosome component rather than three. The current production release
and unchanged painter reproduce **334 founder-allele mismatches / 1,552,807
called cells**, with 443 uncalled cells out of 1,553,250. The earlier fragmented
result had 1,509 mismatches / 1,552,641 calls. Reverse truth-to-panel
errors/missing decrease from 1,920 to 777. These are founder reconstruction
metrics, not sample-painting phase-switch counts. Typed painting reload and final
release checkpoint replay both pass.

The formerly isolated block has six final paths with 200 calls each, using
five distinct local sequences. Existing joint completion fills the retained
panel's one remaining unknown; its pre-fill inference snapshot stays unknown.
The six true local patterns are not all recovered: the reverse local distance
is two allele cells both before and after this change. Thus the successful
bridge is not evidence that all rare local variation has been recovered.

Joining the chromosome initially exposed a long-phase error. The existing
pre-count paired suffix search proposed a much better full-objective phase but
its extra nondecreasing-genotype-fit veto rejected it. The narrowly changed
pre-count policy preserves every site's called/missing allele multiset and
accepts improvement of the existing full objective. Allele-changing moves,
count-reduction refits and final bounded intervals retain their genotype-fit
guards. Truth is used only after inference. The integrated pre-count pass
reproduces the isolated prototype exactly; on five other cached chromosome
inputs, all called arrays and all atomic source-provenance arrays are unchanged.
The default guarded call also reproduces the previous chr13 pre-count result.

The completed bounded controls give the following founder-level results.
Errors are nearest-true-founder mismatches for each assembled row; the reverse
metric separately measures missing or misrepresented true variation.

| Seed / chromosome | Components, before → after | Allele errors, before → after | Called cells, before → after | Reverse errors/missing, before → after |
| --- | --- | --- | --- | --- |
| 2000 / chr13 | 1 → 1 | 15 → 14 | 1,552,809 → 1,552,809 | 456 → 455 |
| 2005 / chr23 | 1 → 1 | 496 → 450 | 2,951,836 → 2,951,531 | 3,012 → 3,271 |
| 2006 / chr1 | 1 → 1 | 194 → 275 | 1,974,707 → 1,974,917 | 1,833 → 1,704 |
| 2006 / chr6 | 9 → 1 | 443 → 544 | 2,445,303 → 2,446,566 | 2,776 → 2,812 |
| 2006 / chr11 | 5 → 1 | 83 → 83 | 2,341,221 → 2,341,622 | 3,572 → 3,567 |
| 2006 / chr13 | 3 → 1 | 1,509 → 334 | 1,552,641 → 1,552,807 | 1,920 → 777 |
| 2006 / chr20 | 3 → 1 | 4,094 → 1,927 | 1,769,097 → 1,769,283 | 4,653 → 2,500 |
| 2006 / chr22 | 3 → 1 | 12 → 3 | 2,398,215 → 2,398,409 | 145 → 142 |

All completed candidates have six rows in one component, with successful
unchanged painting, typed painting validation and checkpoint resume. Chr13 uses the
current production phase-only policy and canonical release replay. The five
original controls have identical pre-count allele and atomic-provenance arrays
under that policy, with unchanged downstream numerical functions; chr20 and
chr22 run the complete latest-source path.

These are not uniform accuracy improvements: chr6 and chr1 gain calls but add
101 and 81 mismatches, respectively; chr23 loses 305 calls and its reverse
metric worsens despite fewer called errors. Joining fragments also makes
long-range orientation errors visible to chromosome-level scoring, which
previously matched each fragment independently. Local 200-SNP errors for chr6
increase from 43 to 56, whereas chr1 decreases from 33 to 29.

Seed2006 chr11's completed final refinement and painting preserve its 83
mismatches while adding 401 calls, with 3,484 unknown cells remaining.
Its cached-input run took 10,253 seconds; the difficult founder-count proposal
dominated, despite distributing other independent proposals to freed nodes.
Seed2006 chr20 also finishes as one six-row component: 1,927 errors over
1,769,283 calls, with 573 unknown cells. Its local 200-SNP errors decrease
34 → 31, although the local reverse errors/missing increase 832 → 862.
All five originally fragmented chromosomes (6, 11, 13, 20, 22) are now joined.
Across all eight comparisons, called errors decrease 6,846 → 3,630 over
16,985,829 → 16,987,944 calls; reverse errors/missing decrease 18,367 → 15,228.
The final chr20 continuation took 196 seconds after expensive count proposals
were cached; that is not its full runtime or a fresh-run timing comparison.
The completed chr22 comparison uses the latest production source
through both feedback rounds, assembly, refinement and painting; it took
643 seconds on 76 allocated cores, excluding cached discovery. Its successful
join adds 194 calls and reduces both forward and reverse errors.
No fresh simulation, whole-genome pedigree–recombination run, high-N sweep or calibrated linkage
confidence study is claimed. These checks establish targeted behavior and
implementation agreement, not whole-pipeline biological accuracy.

## Painting and pedigree inference

Painting comparisons retain sample/site order, component manifests, called
alleles, uncertainty, resets and raw-evidence identities. Additional checks
cover float32/64, sparse/read-only arrays, pooled/unknown states, invalid raw
mass, missing samples and fragmented components exhausting the cache budget.

painting's optional packed emission cache and pedigree's child-tiled common-emission cache
are distinct performance mechanisms. Neither replaces uncertain alleles by
calls or changes candidate eligibility. Absent or oversized caches use the
same underlying scorer. Source transitions remain ordered.

Full metadata-free pedigree checks retain all seven published result tables,
including Tier A/B, support/confidence diagnostics and evidence summaries.
Independent simulation truth is read only after inference. Both the q16
seed402 check and the latest seed401 check recovered:

- **320/320 exact configurations:** 20 M0 roots and 300 M2 parental pairs.
- **600 correct edges**, with no missing or extra edges.
- Identical discrete results to their accepted input-matched references;
  the latest floating diagnostics match their stated numerical tolerance.

The latest seed401 check used new packed chr3/chr16 paintings and the uncached
path on the other 20 chromosomes. Existing L4 inputs were held fixed.
Completed global-checkpoint resume was tested separately without rescoring.

These are specific simulated-design results. M0 means zero parents among the
observed samples, not proof that an individual is a biological founder.
Same-depth and missing-parent designs remain limitations of direction inference.
Internal support and bootstrap fractions are not calibrated correctness
probabilities. Real cichlid cohort labels do not establish individual parentage.

### Conditional ancestry-depth resampling at N=80

At 5x simulated coverage with 20 F1, 30 F2 and 30 F3 samples, full 22-chromosome,
metadata-free decision replays gave the following exact Tier-B configurations:

| Seed | Previously reselecting mixture dimension | Conditional dimension |
| --- | ---: | ---: |
| 2000 | 80/80 | 80/80 |
| 2001 | 75/80 | 80/80 |
| 2002 | 80/80 | 80/80 |
| 2003 | 50/80 | 80/80 |
| 2004, untouched validation | 51/80 | 80/80 |
| 2005, untouched validation | 77/80 | 80/80 |

Each conditional result retained all 20 roots and all 60 exact parental pairs:
120 correct edges, no extras or missing edges. The full-data graph and mixture
selection were unchanged. No true generation count, parent identities or
sample metadata entered fitting; ground truth was used only after saving the
inferred results. The native implementation reproduced the four prototype
replays and passed serial/forkserver, uninformative-data and cached-resume
checks. Seeds 2004 and 2005 were evaluated after selecting the method.

This is evidence for conditional model stability in these simulated designs,
not calibration of correctness probabilities or proof across arbitrary
pedigrees. In particular it does not remove same-depth/missing-parent
limitations. No higher-N rerun was used for this change. Separate N=80, 10/30/40-design
checks at 5x also retained the exact full-data graphs. Conditional Tier-B
recovery was 80/80 for seeds 2010 and 2011, versus 0/80 and 50/80 previously;
each conditional result had 10 roots, 70 exact parent pairs and 140 correct
edges, with none missing or extra. These are pedigree decision-replay results,
not a claim that downstream phase is unchanged after releasing more families.

Standard pedigree-to-recombination replays on the unchanged, held-out seed2004/2005 assemblies
and typed paintings reproduced the exact pedigrees and completed all 22
chromosomes. Final sample phase-switch errors fell from 334 to 172 (2004) and
186 to 171 (2005). Component-aligned sample allele errors fell from 11,358,562
to 426,545 and from 1,551,077 to 427,747 respectively. These are sample-phase
metrics, not reconstructed-founder allele errors. Called-allele totals remained
1,432,247,927 and 1,431,929,896 out of 1,433,096,320, and genotype-error totals
remained 109,486 and 107,975. This supports the downstream benefit of releasing
the correctly inferred families without changing genotypes or coverage.

The corresponding standard 22-chromosome downstream replays for the additional
10/30/40 design completed as well: switches fell from 463 to 149 (seed2010)
and from 371 to 196 (seed2011). Component-aligned sample allele errors fell
from 19,595,833 to 639,973 and from 9,297,924 to 803,119. Called alleles stayed
at 1,412,951,998 and 1,414,401,014 out of 1,433,096,320; genotype errors stayed
at 147,197 and 133,058. Thus these changes improve phase conditional on the
available calls, not the lower coverage of these two assemblies.

A matched seed2003 control also completed standard pedigree–recombination on the unchanged
assembly: 80/80 exact configurations, 120 correct edges and no extras. Switches
fell from 330 to 155 and component-aligned sample allele errors from 11,244,419
to 610,831. Called alleles (1,426,176,454) and genotype errors (128,427) were
unchanged. This isolates the pedigree-resampling contribution; it does not
include the separate provisional founder-assembly changes.

## Expanded founder search at N80 and 5x

The default final refiner now combines beam/dual search, guarded paired suffix
moves, bounded one-founder deletion/refitting, completed exact-flank window
searches, and guarded paired intervals. The local discovery and feedback models,
genotype likelihoods, switch penalties and calling thresholds were not retuned
for N=80. No generation labels, true founder count or true pedigree enter these
searches. The normal simulation defaults were not changed.

The following 22-chromosome comparisons held discovery and hierarchy inputs
fixed within each seed. “Before” is the preceding dual-escape production
refiner. Errors match each reconstructed haplotype to one truth founder over
its whole component; they are **founder allele errors**, not sample phase
switches. Called denominators can change when redundant paths are removed or
different partially observed local rows are selected.

| Design | Seed | Founder errors before → after | Called founder alleles before → after |
| --- | ---: | ---: | ---: |
| 20/30/30 | 2000 | 5,860 → 3,144 | 53,714,690 → 53,714,690 |
| 20/30/30 | 2001 | 5,973 → 5,408 | 53,725,935 → 53,726,386 |
| 20/30/30 | 2002 | 33,959 → 28,788 | 53,708,943 → 53,708,958 |
| 20/30/30 | 2003 | 229,018 → 34,649 | 54,127,745 → 53,687,190 |
| 20/30/30 | 2004 | 3,726 → 2,847 | 53,728,311 → 53,728,281 |
| 20/30/30 | 2005 | 17,740 → 1,886 | 53,724,930 → 53,724,899 |
| 10/30/40 | 2010 | 619,476 → 521,073 | 53,990,325 → 53,478,174 |
| 10/30/40 | 2011 | 628,676 → 320,935 | 54,556,693 → 53,671,358 |

Across the six primary seeds, founder errors fell 296,276 → 76,722
(74.1% fewer); reverse truth-to-panel errors/missing fell
368,649 → 200,402. Thus the gain is not simply fewer reported founder rows.
Reverse errors/missing in ancestry carried by at least two sampled root
lineages fell 242,771 → 90,888. Lineage support is an evaluation diagnostic,
not proof that a tract is statistically identifiable. Unsupported ancestry
was not used as a target for forced reconstruction.

Every genome improves in aggregate, but not every chromosome does.
For example, seed2002 chr3 has 432 additional founder errors; secondary
seed2011 chr10 has 4,313 additional errors in multiply sampled ancestry.
The 10/30/40 design remains materially harder, despite its correct pedigree.
These development-seed comparisons support the mechanism, not uniform
accuracy, global optimality, or a claim that all remaining ancestry is
recoverable.

A fresh primary seed2006 used the ordinary simulation/discovery/feedback and
hierarchy CLI, followed by the same frozen candidate's native final release,
typed painting rebuilding and standard pedigree–recombination. It finished with **13,086 errors /
53,711,112 called founder alleles** (243.6 per million), and 40,894 reverse
truth-to-panel errors/missing. It is not assigned a before/after comparison
against the older dual-only default, which was not run on this seed.

All nine final pipelines recovered **80/80 exact pedigree configurations**:
20 roots, 60 exact parent pairs and 120 edges for each primary seed;
10 roots, 70 exact parent pairs and 140 edges for each secondary seed.
There were no missing or extra parent edges. All 22 chromosomes completed
family refinement, final phase and conditional maps.

| Seed | Final called sample alleles / 1,433,096,320 | Genotype errors | Final sample switch errors |
| ---: | ---: | ---: | ---: |
| 2000 | 1,432,069,069 | 110,857 | 188 |
| 2001 | 1,432,028,139 | 106,996 | 129 |
| 2002 | 1,431,690,483 | 109,127 | 163 |
| 2003 | 1,431,849,349 | 129,585 | 154 |
| 2004 | 1,432,248,237 | 109,427 | 172 |
| 2005 | 1,432,011,021 | 106,017 | 168 |
| 2006 | 1,431,525,586 | 112,539 | 154 |
| 2010 | 1,424,163,119 | 147,225 | 147 |
| 2011 | 1,425,839,280 | 131,986 | 196 |

Matched downstream comparisons also expose trade-offs. Relative to the
completed stable/tie-window search without the later upper-ranked and paired
interval passes, seed2001 has 298 more genotype errors on identical calls;
seed2002 has 769 more. Seed2011 has one additional switch on identical
genotype-correct comparisons (193 → 194), and its component-aligned sample
allele errors increase 717,280 → 740,672 on common calls, despite fewer
genotype and founder errors. In contrast, seed2005's matched genotype errors
fall 107,900 → 106,013 with the same 168 switches. Acceptance prioritizes the
substantial founder/pedigree gains while retaining these measured costs;
improved founder assembly is not automatically improved sample phase.

The final paired-interval addition alone changed two of the 176 cached
chromosomes: seed2005 chr23, 11,544 → 496 founder errors, and seed2011 chr1,
31,058 → 4,580. It introduced no founder-error regression against its
immediate predecessor on these controls. It reduced their genome-wide matched
sample switches from 178 → 168 and 200 → 196, respectively.

Canonical integration matched the frozen candidate's executable syntax trees
(formatting and one explanatory module docstring aside). It passed 11,928
independent interval-boundary/reversal/neutral/ragged score checks, 432
exhaustive count-bound comparisons, 240 window-bound/proposal comparisons,
and three missing-data/provenance/component-boundary/resume fixtures.
Configuration routing exercised the caller's complexity scale and the
disabled route. Every cached native chromosome reproduced its prototype,
rebuilt typed painting, and resumed exactly.

No higher-N simulation was run for this change. No N=80-specific inference
branch or threshold was added; preservation at larger N has not been
empirically re-established by these tests. The extra fresh 10/30/40 seed2012
is separate, still provisional until all of its checkpoints and evaluations
complete. Full validation artifacts remain in ignored work storage.

## Family refinement and final phase

Family phase begins phase assessment after 20 family iterations and requires five
consecutive unchanged final-phase checks unless the family fit converges sooner.
It preserves called genotypes and missingness and does not release full
marginal-posterior tensors. Phase stability is not latent convergence.

Full 22-chromosome comparisons on frozen earlier q4 painting/pedigree/raw inputs showed:

| Seed | Switch errors: full fit → phase-focused | Component-aligned allele errors: full fit → phase-focused |
| --- | ---: | ---: |
| 400 | 1,428 → 1,425 | 1,710,466 → 1,694,466 |
| 401 | 1,594 → 1,596 | 1,678,251 → 1,669,791 |
| 402 | 1,240 → 1,238 | 1,643,154 → 1,642,200 |

Called coverage, genotype counts/errors and comparison denominators matched.
Genotype-error totals were 639,617 / 694,009 / 655,561. Literal phase arrays
changed at 46,010 / 35,168 / 33,120 entries. Comparable accuracy is not exact
equivalence or uniform improvement. Aligned errors permit one strand swap per
sample/component and include genotype errors; switch counts do not bridge
invalid heterozygote comparisons.

A 60,000-marker chr16 control masking 30% of scaffold alleles retained five
switches and 8,179 aligned errors in both fits. A control with missing-read
tracts and one-track gaps gave 31/8,225 after phase-focused stopping versus
29/8,121 after extended fitting. Coverage/genotype errors matched. This is a
small measured stopping-policy trade-off, not numerical roundoff.

Canonical integration retained all 312 arrays in 24 coupled continuation cases
and matched the approved full seed401 results plus six chromosome controls.
Interrupted and completed resume, an interruption before phase assessment, and
refusal to publish at an unstable cap were checked. Later exact reuse passed
all 22 seed401 chromosome comparisons; the newest dirty-tile optimization
retained all final-phase and stability fields on chr16 and chr3.

A failure-only damping retry was validated using the actual exhausted
seed408, N=40 (20/10/10), 5x chr15 checkpoint. Its original solve reached 520
iterations without stable called phase. A cold retry at damping 0.25 stabilized
after 25 iterations and exactly matched the independently computed lower-damping
allele calls and phase map. The original failed checkpoint was retained.
Cold ordinary-damping controls on seed407 with the same N/design chr15 and
seed404, N=320, 5x chr1 exactly matched their previously accepted allele calls
and phase maps; neither used a retry. All three passed resume checks, including
the exhausted first attempt. These are targeted stability/equivalence checks,
not a claim that the latent family posterior converged or that lower damping
uniformly improves phase accuracy.

## Conditional recombination maps

Recombination consumes final phase without feeding upstream. Shared-family orientation
corrections are distinct from independent crossovers. Expected rate, called
crossover intervals and observable meiosis exposure are reported separately.

Phase-focused stopping can change conditional maps slightly. In the original
seed402 chr7 comparison, called crossovers changed 1,724→1,726 and expected
counts 1,837.899→1,837.979. Chr9 calls remained 925; expected counts changed
986.801→987.765. Maximum rate-bin differences were 0.01555/0.09923 cM/Mb.
Those changes are not evidence of improved map accuracy.

The latest indexed shared-orientation update preserved accepted orientation
moves, crossover intervals and coverage on all 22 fixed seed401 final-phase
inputs. Floating map arrays matched rtol=2e-9, atol=2e-8. Small checks covered
overlapping parent/child flips, asymmetric priors, missing calls, gap/component
resets and variable maps. Native-cache reload and output/resume were also checked.

A 100-Mb zero-map case exposed non-finite unguarded transfer products. The
public path now retains stable streaming for disabled artifact processes or
flat genetic-map stretches and matches the streaming reference. The fallback
changes computation, not the statistical map model.

## Reproducing and interpreting validation

The supported validation route is the simulation/evaluation CLI, not a separate
test-only pipeline:

```bash
python run.py simulate --config configs/simulation.toml --seed 400
python run.py evaluate --output work/runs/seed_400
```

Run on allocated resources and use a separate output root when comparing
scientific settings. Evaluation uses cached truth only after reconstruction.
Retain stage identities, actual configurations, raw and matched denominators,
missingness, error counts and completion state. A completion marker or an
empty log alone does not prove correctness.

The package imports, all CLI help paths, TOML parsing and documentation links
are checked before commit preparation. These are wiring checks, not another
scientific validation run. Local detailed reports/results remain in ignored
work storage; development source and full experiment diaries are archived
recoverably rather than included in the public package.

Tropheops and AstCal readable handoffs were preserved byte-for-byte. They have
no established individual-level trio ground truth; preservation is not a
real-data accuracy test. See each deliverable's README for historical caveats.

## Pedigree cache separation and release diagnostics — 20 September 2026

The preparation/genetic-scoring/decision cache split, readable pedigree
explanations, founder-support/search reports and general AD-BCF command were
validated without changing inference defaults:

- Six saved-genome controls, **820 sample rows**, retained identical Tier A,
  Tier B and complete pedigree tables. Controls covered N80/N160, the N300
  deeper-generation case, canonical parent withholding and ragged backcrosses.
  Full-fit reported score contributions sum to the actual adjusted scores.
- The instrumented bounded assembly selector retained exactly the same paths,
  scores and accepted edits as the pre-change implementation, at the production
  proposal budget and an intentionally small budget exercising truncation.
- A real indexed BCF fixture (40 samples, three chromosomes, 4,800 SNPs,
  approximately 5x, missing AD and sample tracts) completed the public command
  from discovery through final phase and map output. Input read counts, raw
  likelihoods and observation masks round-trip exactly. Founder calls and
  final phase are unchanged by the added reports. This short fixture released
  no parent edges, so it is **integration validation, not map or pedigree
  accuracy validation**. A warm-code 112-core run took 35.5 seconds; a full
  resume took 0.6 seconds after imports. These are not whole-genome estimates.
- Actual checkpoint round trips confirm that direction-policy changes reuse
  preparation and genetic scoring; eligibility changes reuse preparation but
  rescore. Unchanged Tier-B relationships reuse family phase. Previous decisions remain
  available in versioned checkpoints.
- Focused fixtures cover component-local missing-founder support, counting
  each observed homozygous carrier once, chronology/eligibility parsing and
  score tracing without numerical changes. Syntax and whitespace checks pass.

An initial integration canary exposed a stale preparation-identity field lookup,
which was repaired before the final clean run. The successful CLI test emitted
non-fatal loky semaphore-tracker shutdown warnings; both full run and resume
exited successfully with all final checkpoints complete. Existing accepted
campaign results were not rewritten. Diagnostic support remains model-dependent
and is not presented as calibrated probability or new scientific validation.

## Balanced transmission evidence and missing parents — 20 September 2026

The seed6200 missing-individual control exposed a residual uncertainty problem:
three fish had a strongly supported observed parent but unstable M1/M2 state,
and one had unstable M0 support. Their missing biological parents had observed
relatives that competed as parent candidates. The existing finite direction,
reciprocal-family and short-ancestry-path corrections were active. This was
a distinct scoring limitation, not an absent direction correction.

Ablations showed that the information exponent inside the transmission forward
algorithm could conceal distinguishing genetic contradictions. Full-strength
scoring alone fixed this control but lost one Tier-B call under read stress;
it was not adopted. The default now uses the fixed equal-weight log-score pool
described in [Methods](methods.md#pedigree). No release thresholds, state priors,
eligibility rules, error rates or family-message settings were relaxed.

All following counts are exact **Tier-B configurations**, including M0/M1/M2
state and the identities of every observed parent. Candidate shortlists were
rebuilt with the proposed score; the results are not limited to a frozen
shortlist.

| Seed6200, 5× control | Samples | Previous | Balanced |
|---|---:|---:|---:|
| Ordinary 20/30/30 cross, reduced density | 80 | 80 | 80 |
| Backcross, reduced density | 80 | 80 | 80 |
| Randomly observed 80% of a 200-fish cross, reduced density | 160 | 156 | 160 |
| Read stress, reduced density | 80 | 80 | 80 |
| Observed F4–F6 only, reduced density | 80 | 77 | 80 |
| Missing-individual cross, full density | 160 | 160 | 160 |
| Backcross, full density | 80 | 80 | 80 |

Reduced density means every tenth template SNP at its original coordinate,
across all 22 physical contigs: 895,695 sites rather than approximately
8.96 million. These seed6200 designs are related development controls, not
independent seeds. Read stress combines depth CV 0.6, 10% contiguous marker
dropout, heterozygote alternate-read probability 0.6 and generating read-error
probability 0.04, with inference assuming 0.02. The original complete pipelines
were run from discovery onward; scoring comparisons reuse their frozen
founders, painting and prepared evidence. The new scoring model was not used
to regenerate their downstream family phase or recombination maps.

The reduced missing-individual result recovers all **234 true observed-parent
edges with no extras**, and all four previously unresolved configurations.
Each has 1,000/1,000 chromosome-bootstrap state/configuration agreement and
22/22 leave-one-chromosome-out agreement. These are internal stability measures,
not calibrated probabilities of correctness. The 147 founder allele errors
among 5,374,026 called alleles are unchanged: this intervention changes
pedigree scoring, not assembly.

Further controls held out when choosing the equal weight passed with complete
candidate rescreening: full-density seeds2003 (80/80), 2010 (80/80), 4000
(160/160), 4006 (160/160), and the previously difficult F4–F6 seed5001
(300/300, including all 100 M0 roots and all 400 edges). They are independent
simulation seeds, but not seeds unused in all earlier project development.
There were no extra or missing released edges in these controls.

Additional random 20% sample withholding *before* candidate screening gave
64/64, 64/64, 128/128 and 128/128 for seeds2003, 2010, 4000 and 4006,
respectively, under both the previous and balanced models. These tests freeze
the original upstream assembly/painting; they validate pedigree candidate
withholding, not a new end-to-end missing-parent reconstruction.

The trade-offs are explicit. Strict Tier-A coverage in the noisy-read control
falls from 80 to 77, while all 80 remain exact in the primary Tier-B product.
Full-strength weights of 25% and 50% retained all five reduced-control
pedigrees; 75% lost the stress call. The default retains the originally tested
equal weight, without per-sample selection. Matched full-density withholding
tests took about 1.5–1.6 times as long for genetic scoring plus pedigree
decisions; this is not a whole-pipeline slowdown factor. Warm reduced-density
score/decision times were about 47 seconds for N160 and 17 seconds for N80
on 112 allocated CPUs, excluding prepared-input loading. The larger N300
F4–F6 rescreen took about 13 minutes for scoring plus decisions on the same
CPU allocation.

Focused checks verify missing-child neutrality, missing-parent M2-to-M1
reduction, lower-order score reuse, candidate/sample order, independent
score-array agreement (atol 1e-7, rtol 1e-11), and ordinary versioned-checkpoint
round trips. Preparation identities remain unchanged, score identities change,
and unchanged relationship tables retain their downstream refinement identity.
These finite controls support the change; they do not establish universal
identifiability or calibrated pedigree probabilities.

## Final reciprocal reconciliation — 21 September 2026

The default now freezes the joint ancestry-path correction and performs a
final reciprocal solve on the path-adjusted scores, replacing rather than
duplicating the earlier reciprocal term. It does not recompute the path
factor from those final messages. This is a bounded composite-model change;
genetic scores, candidate panels, priors and release thresholds are unchanged.

The motivating seed7009 control has 320 observed samples, 5x reads, all 22
chromosomes, pedigrees extending to F10, 50% backcrossing and 20% randomly
withheld individuals. Later path evidence changed which descendants competed
as parents, leaving the earlier reciprocal messages inconsistent with the
final scores. Increasing only the original message-pass count did not fix it.

| Seed7009 quantity | Before reconciliation | Reconciled default |
| --- | ---: | ---: |
| Correct local configurations before DAG cleanup | 319/320 | 320/320 |
| Complete full-data DAG configurations | 320/320 | 320/320 |
| Exact Tier-B configurations | 316/320 | 318/320 |
| Exact Tier-A configurations | 315/320 | 316/320 |
| Correct Tier-B edges / true edges | 474/476 | 475/476 |
| Extra Tier-B edges | 0 | 0 |

The canonical production API, normal forkserver workers, 1,000 fixed-seed
chromosome bootstraps and 22 LOCO fits reproduce the serial prototype's
relationship tables and checked support columns. The family-phase edge reader
accepts the 475 correct released edges. The two remaining Tier-B abstentions
are a separate missing-parent/relative ambiguity, not resolved by this change:
one observed parent remains supported in every resample but M1/M2 count is
unstable; another fish has unstable M0 support when distinguishing chromosomes
are omitted.

Full-data production checks cover 38 cached seeds (7,420 sample evaluations),
all with correct local and graph configurations. The other 37 seeds preserve
their calls. The prototype also preserved 1,980/1,980 exact Tier-A and Tier-B
configurations across eight matched 100-bootstrap/22-LOCO controls spanning
N80/N160/N320, missing individuals, backcrosses and all-M0 cohorts. These are
cached evidence replays, not 38 new end-to-end simulations.

Production integration additionally preserved all 1,980 exact calls at both
tiers in those eight controls, using 1,000 bootstraps for seed5001 and 100 for
the other seven. Full and partial caller-chronology backcross controls each
retain 40/40 at both tiers with 1,000 bootstraps. Every integration case used
two normal forkserver workers and passed its checkpoint round trip and
family-phase edge-consumer checks. These are interface/decision validations,
not reruns of downstream chromosome phasing.

Focused score fixtures check missing/identity-ineligible rows, partial/full
chronology, component ablations, unchanged supporting path factors and the
final diagnostic score decomposition. Preparation/genetic-score identities
are unchanged by this integration; decision identities change, and family
phase is bound to the resulting relationship table. Existing campaign results
are preserved. No whole-genome downstream phase or recombination rerun is
claimed. The model remains approximate and its support is not a calibrated
parenthood probability.

## Marginal parent-edge release and fixed-genome exclusion prototype (21 September 2026)

The canonical partial-parent output now separates support for an individual
edge from support for an exact M0/M1/M2 configuration. It uses the unchanged
parent-edge bootstrap/LOCO cutoffs in both local and graph views, requires
agreement with the full-data acyclic graph, and retains count uncertainty.
Eleven normal API replays preserve the existing exact primary calls. Only
F9_33 -> F10_29 is newly retained in the seed7009 control, giving family
conditioning all 476 true edges with no extras, while the exact Tier-B table
still contains 318/320 configurations.

Both partial tables are checkpointed and exported. Family refinement and
recombination consume the same explicitly supported-edge table; changing a
support flag changes the downstream relationship identity. Focused checks
verify checkpoint/CSV round trips, omission of an unsupported co-parent and
identical missing-aware family messages to an equivalent one-known-parent
fixture. These are interface checks, not new full-genome phase evaluations.

A separate, **non-default prototype** resolves the two seed7009 count
abstentions using raw genotype likelihoods on the fixed observed genome.
It tests every alternative parent, allows a 1% arbitrary genotype-replacement
channel, averages pre-specified betting fractions/windows, and applies a
multiplicity bound using at most two true direct parents per individual.
It releases M0 only when all sampled alternatives are excluded; M1 additionally
requires the sole nonexcluded parent to have existing marginal edge support
and agree with the full-data graph. Chromosome-bootstrap fractions are retained
unchanged, not reinterpreted as biological probabilities.

This prototype changes seed7009 from 318/320 to 320/320 exact configurations,
with 476/476 true edges and no extras. Across thirteen 22-chromosome cached
datasets (3,520 samples, 5,847 true edges), no true parent was excluded and no
other call changed. Eight seeds were held out from construction:
7000, 7002, 7006, 7007, 7011, 7012, 7013 and 7014. The read-stress seed7010
remains 133/160 exact; this method does not repair its separate errors and
abstentions. Total exact calls in these controls are 3,491 -> 3,493.
These tests reuse raw-GL/painting checkpoints, not new assembly runs.

The exclusion argument is conditional on calibrated read likelihoods and
conditional read-error independence; inherited SNP genotypes may be linked.
Twenty-seven enumerated read/genotype/error-channel checks verify its local
bound. Correlated mapping errors, data-dependent ascertainment and other
observation-model violations remain important limitations. Compatibility alone
does not determine relationship direction. The exact-call certificate is not
yet integrated as the production default.

## Family branch marginal projection (23 September 2026)

Live sampling of the read-stressed seed7010 family solve found approximately
20 of 45 wall-clock seconds in the final serial NumPy branch-marginal reductions.
The replacement fuses the same two four-state reductions in a marker-parallel
Numba kernel. It retains float32 intermediate rounding; no likelihood, prior,
damping, iteration limit, phase-release criterion or output schema changes.

Eight constructed multi-generation family fixtures, each with an active branch
cluster and 30 message iterations, had exactly identical messages, residuals
and phase outputs. Float32/float64 chromosome-sized 500,000-marker projections
also matched exactly. On 38 allocated CPUs the reduction-only medians were
approximately 97x and 78x faster respectively. These are **kernel timings**, not
whole-phase or pipeline speedups. The oscillating solver can still require many
iterations; faster projection does not establish convergence or phase accuracy.
Existing frozen-code comparison runs retain their original implementation.

The coupled five-node factor sweep also reuses small work arrays across
256-marker computational tiles and uses the existing scalar four-state cavity
normalization. Every marker and selector state remains; the sequential update
order between overlapping family branches is unchanged. Both intermediate-parent
slots matched exactly in 100,000-marker kernel comparisons, and twelve 60-iteration
multi-generation fixtures retained identical phase decisions. At nine allocated
CPUs the production kernel took approximately 25% less time (1.33–1.35x speedup).
This is a kernel result, not a whole-stage timing or a convergence improvement.
