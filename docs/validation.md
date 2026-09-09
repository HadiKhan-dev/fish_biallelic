# Validation of the canonical package

The September 2026 package reorganization preserves the accepted scientific
models. It changes module layout, supported entry points and checkpoint names;
old flat-module pickle compatibility is deliberately not retained.

## Reorganization checks

| Check | Result |
| --- | --- |
| 22-chromosome seed-72 pedigree evidence | All seven scientific/support tables exactly match the pre-reorganization computation; 20 M0 roots and 300 M2 parent pairs |
| L1–L4 plus component painting | Identical discrete calls and posterior arrays for a 16-block, 64-sample, four-founder fixture with missing observations |
| 200-SNP discovery | Exact results for ordinary low-depth data and a targeted rare-founder missing tract |
| Joint family refinement | Exact calls, probabilities and messages on a five-individual, two-generation missing-data fixture |
| Genotype-preserving phase polishing | Exact phase paths/selectors on a missing-data pedigree fixture |
| Shared-family recombination inference | Exact numerical outputs on a shared parental orientation-error fixture |
| Package imports | All 80 importable modules/packages load after removal of flat source files; internal module-attribute references resolve |
| New CLI, full workflow | A three-short-chromosome, 320-sample wiring canary completes through T12 |
| Completed checkpoint restart | The canary resumes through T12 in 4.65 seconds without recomputing completed scientific stages |

The cached seed-72 comparison checks the inference engine on identical existing
evidence, not a new seed-72 assembly. The short CLI canary is deliberately too
small for useful pedigree inference and is **not** evidence of genome-wide
accuracy. Nontrivial family/meiosis behavior is exercised separately by the
family and recombination fixtures. Runtime/diagnostic timestamps were excluded
from numerical fixture comparisons; numerical arrays and decisions were exact.

The full cached pedigree computation took approximately 25 seconds before
reorganization and 27 seconds in the first packaged comparison on the same
76-CPU allocation. This is an implementation-regression check, not a controlled
end-to-end speed benchmark. Compilation, I/O and startup matter at different
scales.

A live assembly check caught a package-move runtime regression: automatic
Numba caching was still scoped to `core/`, excluding kernels in other
subpackages. The scope now covers the whole package. A fresh L1–L4/painting
fixture after that correction again matched every numerical/discrete output;
later levels reused compiled kernels. Affected full assembly attempts were
restarted with fresh source identities while their complete simulated inputs
and 22-chromosome block-discovery checkpoints were retained and reused.

## Shared linker at every assembly level

The default now uses the same distance-aware normal/burst HMM at L1-L4,
with a maximum of 20 fitting iterations per gap (convergence can stop earlier).
This is a deliberate change to the L1 model, separate from the model-preserving
package reorganization above. L1 grouping and its unlimited beam-gap rule are
unchanged. The specialized L1 implementation has been removed.

The comparison used cached chr16 inputs from three independent 5x seeds,
320 samples and six simulated founders, on 76-core Ice Lake nodes. Old-L1
and shared-20 L1 timings are fresh measurements on the same node for each seed.

| Seed | Old L1 seconds | Shared-20 L1 seconds | Old final founder mismatches | Shared final founder mismatches |
| --- | ---: | ---: | ---: | ---: |
| 400 | 101.57 | 176.00 | 281 | 309 |
| 401 | 103.44 | 179.71 | 271 | 307 |
| 402 | 103.14 | 176.28 | 357 | 372 |

L1 was 71-74% slower. Final L4 output had 909 versus 988 closest-truth
founder-allele mismatches across 6,579,146 called founder cells: 79 additional
mismatches, an 8.7% increase in the mismatch count and approximately 0.0012
percentage points in the error rate. This is a small absolute accuracy
regression, not an accuracy improvement or a sample phase-switch measurement.

Final called coverage was unchanged at 99.99015%, 99.97028% and 99.95188%.
Component and haplotype-row counts were unchanged at L4. Reverse truth-to-output
comparisons that count missing/omitted sequence also became slightly worse;
the apparent similarity is not achieved by reducing called coverage.
A targeted dropout fixture retained the same disconnected missing block and
excluded samples with no evidence from informative carrier counts.

The architectural benefit is one linker implementation and one fitting cap,
removing approximately 1,100 lines of specialized/duplicated code. This result
does not establish genome-wide or real-data accuracy, nor downstream sample
phase-switch equivalence. Existing simulated inputs and discovered blocks
remain useful, but old-model assembly/painting outputs have distinct scientific
checkpoint identities and cannot masquerade as outputs of the new default.

After promotion, the canonical assembly route reproduced all four levels of
the shared-20 experiment exactly on all three seeds, including numerical
arrays, missingness and atomic founder paths. All 82 importable modules and a
two-worker missing-data smoke passed. Fresh production L1 timings were 178.72,
182.29 and 179.88 seconds; the complete assembly call, reusing unchanged
preprocessing, took 281.56-291.31 seconds. Seed 400 resumed preprocessing and
all four hierarchy checkpoints in 13.03 seconds with no hierarchy recomputation
and identical final results. Old-model hierarchy checkpoints were correctly
rejected under the new identity; accepted previous results were preserved.

## Fresh full runs

Seeds 400 onward use the current pipeline, frozen empirical sequence templates,
5× mean depth and 20/100/200 sample cohorts by default. Every completed stage
is checkpointed, including true simulated alleles and raw crossover events.
`python run.py evaluate --output work/runs/seed_400` produces post-inference
pedigree, coverage, phase and conditional map comparisons. Do not infer success
from an empty error log or a completed marker alone.

Real-data handoffs have no established individual-level trio ground truth.
Their exports were checked for exact file preservation, not real-data accuracy.
See each deliverable's README for historical interpretation caveats.
