# Validation and known limitations

The pipeline is evaluated with known-pedigree simulations and fixed-input
implementation comparisons. These answer different questions: numerical
equivalence does not establish biological accuracy, and a correct pedigree does
not imply perfect founder reconstruction or sample phase.

Default simulations use frozen empirical founder templates, 5× mean read depth
and 20/100/200 sample cohorts. The 22 template contigs are chr1–20, chr22 and chr23.
Seeds400–402 are development data; seed401 was held out specifically when
choosing the T11 stopping rule, not from all project development.

## Current acceptance coverage

| Portion | Completed evidence |
| --- | --- |
| Stage1 discovery | All 1,647 seed400 chr1 blocks; repeated batches and 76 approximately 2× seed401 missing-data controls |
| Per-round balanced feedback | 9,287 local blocks in six seed/chromosome cases; full chr12 production parity plus bounded L1–L4/painting and resume checks |
| Pre-L1 founder completion | 5,484 chr16 blocks across seeds400–402; independent 2×/3×/5× crossfits |
| Dense q16 assembly and painting | All 22 seed402 chromosomes plus six controls; Stage1 and painter held fixed |
| Final founder-path refinement | All 22 current balanced-input seed402 and seed403 chromosomes plus seed400–402 controls; geometry, missingness, provenance and resume checks |
| Latest guarded multiscale refinement | All 22 seed403 chromosomes plus six controls; fresh chr4 final assembly, painting and typed T09 replay |
| Painting numerical implementation | All 22 seed402 q16 chromosomes, 148 components and 8,956,852 sites, plus seed400/401 chr16 controls |
| Packed T09 evidence reuse | Full chr3/chr16 repaints, prepared fields and typed checkpoint round trips; memory-limited/uncached controls |
| Metadata-free T10 | Full 22-chromosome seed402 q16 check and a subsequent full seed401 cached/uncached check |
| Phase-focused T11 | All 22 chromosomes of seeds400–402 on frozen earlier q4 inputs; later complete dense q16 seed403 run before multiscale refinement |
| Latest T11 dirty tiles | Two complete seed401 chromosomes; exact final phase/stability fields |
| Latest T12 indexed trials | All 22 seed401 final-phase maps; variable/zero maps, missingness, overlapping flips and reset controls; later complete seed403 run before multiscale refinement |

A fresh seed403 run has now exercised simulation through recombination maps
with the balanced local feedback and final founder refinement. A subsequent
22-chromosome optimization comparison held its Stage1 inputs fixed and reran
feedback, L1-L4, founder refinement, painting, pedigree inference, family phase
refinement and maps. Fixed-input comparisons must not be relabeled as fresh
simulations or fresh block discovery.

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

The measured run reusing Stage1 took 204.03 minutes, excluding final evaluation.
Adding the unchanged baseline input/discovery durations gives a **252.53-minute
fresh-run estimate**, versus 282.87 measured baseline minutes (10.7% less).
This is not a measured fresh-seed optimized runtime, a 112-core measurement,
or evidence that every stage improved. Node differences, filesystem throughput
and native-cache warmth affect elapsed times. The single-read Stage1 I/O change
was not timed by this cached-Stage1 comparison.

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
refinement, unchanged painting and typed T09 replay; refinement is absent from
the L1/L2 context checkpoints. A fresh full chr10 final hierarchy/refiner run
reproduces the result and passes unchanged painting, typed T09 publication and
exact assembly replay. CLI/TOML/environment configuration is checked for all three
workflows, including reuse of local feedback checkpoints when final refinement
is toggled off. The clean implementation matches all 26 paired prototype outputs
exactly; seed401 chr16 is an additional clean-only control.

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
then reaches the same six-error result. Unchanged painting, typed T09 publication
and replay pass; the installed production code also replays all six cached
assembly/refinement phases with identical founder alleles. No Stage1 or feedback
regeneration was needed. T10–T12 were not rerun after this founder change.

On the 76-core node, fresh final assembly plus refinement takes 343.3s, of which
229.1s is refinement; painting takes 15.0s. The stored earlier 76-core chr4 refiner
measurement is 136.9s, so this case costs about 92s more. These are separate
stored-run measurements, not a new full-seed or 112-core benchmark. The full
refiner control at 38 cores takes 287.7s with the same six-error output.

## Missing-data discovery and completion

Focused cases cover missing read cells, wholly unobserved samples, long missing
tracts, duplicate/rare founders, partial founder calls, wildcard assignments,
dtype differences, ties and multiple thread counts. Unknown alleles are not
treated as reference or as independently guessed copies of one shared founder.

All compared Stage1 scientific outputs matched exactly. Across the 5,484
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

## Painting and pedigree inference

Painting comparisons retain sample/site order, component manifests, called
alleles, uncertainty, resets and raw-evidence identities. Additional checks
cover float32/64, sparse/read-only arrays, pooled/unknown states, invalid raw
mass, missing samples and fragmented components exhausting the cache budget.

T09's optional packed emission cache and T10's child-tiled common-emission cache
are distinct performance mechanisms. Neither replaces uncertain alleles by
calls or changes candidate eligibility. Absent or oversized caches use the
same underlying scorer. Source transitions remain ordered.

Full metadata-free T10 checks retain all seven published result tables,
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

## Family refinement and final phase

T11 begins phase assessment after 20 family iterations and requires five
consecutive unchanged final-phase checks unless the family fit converges sooner.
It preserves called genotypes and missingness and does not release full
marginal-posterior tensors. Phase stability is not latent convergence.

Full 22-chromosome comparisons on frozen earlier q4 T09/T10/raw inputs showed:

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

## Conditional recombination maps

T12 consumes final phase without feeding upstream. Shared-family orientation
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
