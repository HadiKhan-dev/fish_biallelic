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
| Pre-L1 founder completion | 5,484 chr16 blocks across seeds400–402; independent 2×/3×/5× crossfits |
| Dense q16 assembly and painting | All 22 seed402 chromosomes plus six controls; Stage1 and painter held fixed |
| Painting numerical implementation | All 22 seed402 q16 chromosomes, 148 components and 8,956,852 sites, plus seed400/401 chr16 controls |
| Packed T09 evidence reuse | Full chr3/chr16 repaints, prepared fields and typed checkpoint round trips; memory-limited/uncached controls |
| Metadata-free T10 | Full 22-chromosome seed402 q16 check and a subsequent full seed401 cached/uncached check |
| Phase-focused T11 | All 22 chromosomes of seeds400–402 on frozen earlier q4 inputs; canonical integration and resume checks |
| Latest T11 dirty tiles | Two complete seed401 chromosomes; exact final phase/stability fields |
| Latest T12 indexed trials | All 22 seed401 final-phase maps; variable/zero maps, missingness, overlapping flips and reset controls |

There has not been a fresh, start-to-end run of every latest component together.
In particular, the full q16 assembly-to-T11/T12 combination remains to be
evaluated. Fixed-input tests deliberately reuse checkpoints instead of
repeating hours of assembly. They must not be relabeled as fresh reconstructions.

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

The latest model-preserving assembly optimizations retained every compared
L1–L4 digest, founder truth metric and unchanged-painter output in their
controlled chromosome checks. The subsequent non-assembly update did not edit
L1–L4 algorithms, transitions, q16 budgets or defaults.

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
