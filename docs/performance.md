# Performance and resource use

The supported pipeline uses missing-aware block discovery, local selection
after each feedback round, L1–L4 assembly with progressive founder refinement, ragged
painting and pedigree inference, family phase correction, and conditional
recombination maps. See [model choices](founder_scaling.md) and
[scientific validation](validation.md).

Timings below use different inputs, CPU counts and cache states. Do not
multiply their speedups or add them into a fresh end-to-end runtime. First-use
compilation, shared-filesystem I/O, founder ambiguity and chromosome length
matter. Configured threads are not a measure of sustained CPU utilization.

## Complete workflow timing

### Fresh seed3000, N320 and 5×

The current implementation completed a fresh 22-chromosome run through T12 on
17 September 2026, using five 76-core Ice Lake allocations and one 66-core
allocation. Measured multi-node elapsed time was **93.67 minutes**, excluding
the subsequent truth evaluation. This includes shared initialization,
synchronization and a deliberate finish-controller restart to distribute T11.

Summing disjoint measured work gives a **single-76-core estimate of about
5 hours 10 minutes** (roughly 5–5¼ hours), not a measured continuous one-node
runtime. The design uses 20/100/200 cohorts and 8,956,852 markers.

| Portion | Estimated 76-core minutes from measured work |
| --- | ---: |
| Shared templates, simulation and read preparation | 10.1 |
| Initial 200-SNP discovery | 39.5 |
| Both feedback rounds, including balanced selection | 88.4 |
| Final L1–L4 assembly and progressive founder refinement | 112.4 |
| Painting computation | 2.6 |
| Pedigree evidence preparation | 3.8 |
| Genome-wide pedigree inference, including prepared-input loading | 15.8 |
| Family refinement and final phase | 19.7 |
| Recombination maps | 3.0 |
| Remaining per-chromosome startup, I/O and bookkeeping | 12.3 |
| **Total, before rounding individual rows** | **307.5** |

The estimate counts shared initialization once, sums independent chromosome
work, and does not add nested timers or multiply six-node elapsed time by six.
It assumes ideal 66-to-76-core scaling only for work executed with 66 cores;
without that normalization the sum is 313.8 minutes. Repeated cold compilation,
process/cache loading, shared-filesystem contention and continuous-process
reuse contribute uncertainty. The interrupted first finish attempt, repeated
cached-stage verification and final truth evaluation are excluded from this
fresh-work estimate.

Independent chromosomes ran with each node's full CPU ceiling and existing
dynamic allocation; this does not imply sustained full utilization. T10 ran
genome-wide on one 76-core node. Once parentage was fixed, a **run-local** T11
scheduler called the unchanged chromosome solver concurrently, retaining the
full stage identity and canonical global completion checks. This scheduler is
not a new production CLI feature. Its 4.35-minute parallel wall time is not the
single-node T11 estimate: the latter sums chromosome solves. A separate chr8
replay matched every final allele call and its 25-iteration stopping point.

See [scientific outcomes and limitations](validation.md#fresh-seed3000-end-to-end-validation).
Detailed timings and frozen source remain in
`.work/seed3000_full_20260917_cFrnkNJ8/`; the readable run report and checkpoints
remain under `work/runs/seed_3000/`, outside Git.

### Earlier workflow measurements

There is still no measured fresh-seed, full-genome runtime for this complete
implementation on a 112-core node. The earlier seed403 comparison took 204.03
minutes from cached Stage1 on 76-core Ice Lake hardware. Adding its unchanged
input/discovery durations gives a 252.53-minute estimate, not a measured fresh
run. It predates the expanded and subsequently accelerated founder refiner.
The [recorded stage breakdown](validation.md#seed403-end-to-end-performance-comparison)
must not be presented as timings of the current full pipeline.

Earlier N80 downstream replays on 76 cores took 244–267 seconds for T10,
259–263 for T11 and 42–43 for T12. They used completed upstream products and
different seeds, not a matched scaling or fresh whole-pipeline experiment.
Discovery, both feedback rounds and final assembly must be included in any
fresh-run measurement.

## Final founder refinement

The complete refiner now runs after each executed level of final L1–L4
assembly, never in the two local-feedback passes. Small components run
concurrently and share a chromosome-derived early proposal resolution;
late levels retain component-specific resolution. The implementation shares
invariant evidence and background scores,
uses compiled beam/dual scans and contiguous emission tables, scores localized
edits with exact flanks, and bounds losing searches. Independent candidates
share the verified CPU budget; freed threads become available at numerical
boundaries. Likelihoods, missing masks, genotype-fit guards and canonical
full-site acceptance are retained.

There are also explicit search approximations. All single-founder deletions
receive cheap repairs, but only the best repaired deletion enters deep search.
The default skips that deep search if every repaired deletion still loses by
more than eight per-founder complexity costs. This is a heuristic, not a safe
mathematical bound. Improving repairs are retained;
`FounderRefinementConfig(count_refit_deficit_multiple=None)` disables the
deficit screen. Large-K interval partners are bounded as documented in the
[work and storage analysis](founder_scaling.md#final-founder-refinement-explicit-work-and-parallelism).

### Progressive final assembly at N80

Matched-input 5× controls on 76-core allocations give the following warm-native
measurements. These include preprocessing, **all final hierarchy levels and
their refinement**, and assembly checkpoint writes. They exclude initial input
loading, truth evaluation, complete-resume verification, discovery, both local
feedback rounds and downstream stages.

| Control | Earlier final-only schedule (s) | Progressive schedule (s) |
| --- | ---: | ---: |
| Seed2005 chr6 | 68.08 | 101.48 |
| Seed2006 chr20 | 64.24 | 99.98 |
| Seed2006 chr3, 1,505,847 markers | 190.90 | 321.39 |

These controls cost approximately 1.5–1.7× the previous final-only schedule,
not four times as much. Seed2006 chr13 takes 81.23 seconds in the progressive
implementation; its stored original comparison includes cold compilation, so
it is not used as a warm speed ratio. Runs were not alternating repetitions;
chr20 uses the same node, while chr6/chr3 compare different 76-core nodes.
First-use native compilation added roughly 170 seconds in these experiments.

The initial naive progressive chr3 run took 970.5 seconds, with approximately
426 GiB of observed parent-process RSS. Sharing early proposal resolution,
parallelizing complete independent components, reusing identical searches and
accelerating metadata validation reduced the current run to 321.4 seconds and
35.2 GiB peak parent RSS, with the same 145 called-allele errors. Parent RSS is
not a node-wide peak-memory measurement. The new primary-preserving tie search
retains the cubic founder-count bound.

The validation used five 76-core allocations and one 66-core allocation
(446 CPUs), with one full-node chromosome stream per allocation. Component and
candidate teams subdivide each node's verified budget and grow at numerical
boundaries. Compilation, metadata checks and I/O still contain serial work;
configured cores must not be interpreted as sustained 100% utilization.
The 66-core timings are kept separate from the table above.

### Earlier final-only N80 warm-cache measurements

These isolate one post-L4 refiner invocation. They predate the progressive
schedule and partial-evidence tie search and are **not** current total
L1–L4-plus-refinement timings.

| Complete final-only refiner replay, 5x, 76 CPUs | Starting implementation (s) | Optimized, unscreened count (s) | Optimized final-only (s) |
| --- | ---: | ---: | ---: |
| Seed2006 chr11 | 114.60 | 47.34 | 20.50–20.62 |
| Seed2006 chr20 | 110.22 | 41.25 | 25.17–25.46 |

Elapsed-time reductions are approximately 82% and 77%. These warm-native-cache
runs include fresh refinement checkpoints but exclude initial input loading,
truth evaluation, discovery, hierarchy and downstream stages. Starting
measurements used the same fixtures and 76-core hardware, not alternating
same-node repetitions. The first integrated chr20 call took 44.05 seconds
including native compilation/cache loading.

Other 76-core controls took 15.95 seconds (seed2006 chr22), 15.03
(seed2000 chr13), 27.54 (seed2006 chr6), 25.78 (seed2005 chr23) and 23.15
(seed2006 chr1). Thus roughly 25 seconds is achievable on the named difficult
fixtures, not a universal limit. Fragmented components and useful deep count
refits can take longer.

The direct-production comparison covers 217 cached chromosomes across ten
N80/5x seeds, representing both 20/30/30 and 10/30/40 cohort designs. All ordered
alleles, missingness, geometry, probabilities and provenance match the frozen
pre-optimization baseline. The screen avoided 163 losing deep refits while
retaining all 11 accepted count reductions. These are empirical comparisons,
not a guarantee for unseen inputs or an accuracy claim for every earlier
algorithm redesign. See [validation scope](validation.md#optimized-founder-refinement).

### Earlier final-only N320 warm-cache measurements

Seven distinct N320/5x chromosomes from seeds404–408 passed field-by-field
comparisons against the frozen predecessor using unchanged cached L4 inputs.
Both versions give 274 founder-allele errors in 20,334,788 called cells, with
592 unknown output cells. This test isolates refinement and deliberately
retains the input component boundaries.

| Same-node warm cache, fresh final-only checkpoints | CPUs | Predecessor (s) | Optimized final-only (s) |
| --- | ---: | ---: | ---: |
| Seed404 chr1 | 76 | 255.96 | 58.48 |
| Seed404 chr11 | 76 | 333.80 | 62.88 |
| Seed405 chr20 | 76 | 337.64 | 47.17 |
| Seed407 chr10 | 76 | 195.87 | 53.45 |
| Seed408 chr19 | 76 | 92.19 | 42.96 |
| Seed404 chr20, fragmented | 66 | 240.87 | 87.84 |

Large seed406 chr3 (1,505,847 markers) took 179.33 seconds on 76 cores with
warm native caches. Separate first-use 112-core runs took 221.79 seconds
optimized versus 642.31 for the predecessor; these are not comparable to the
warm 76-core result as a scaling experiment. Short N320 cases therefore take
roughly 43–63 seconds; the large chromosome still takes about three minutes.

All 27 refinement solves, including paired warm repeats and the additional
76-core chr3 check, passed completed-resume checks. Fourteen optimized outputs
were compared against recomputed predecessor outputs. No discovery or
upstream assembly was rerun in this comparison.

### Cached-Stage1 reconstruction replay

The subsequent eight-fragmented-chromosome check reran both balanced feedback
rounds, partial-founder-aware L1–L4, final refinement and painting. Seven
existing nodes provided 558 CPUs: five with 76, one with 66 and one with 112.
The first available node picked up the eighth chromosome.

From loaded Stage1 inputs through painting/checkpoint output, seven chromosomes
took 372–590 seconds each; large seed406 chr3 took 1,087 seconds on 112 cores.
These include fresh per-node native-cache costs except for the queued chr15
case. They exclude discovery, initial reads, truth evaluation and T10–T12, so
are neither isolated refiner timings nor full fresh-seed timings.

All eight now contain one six-row component, but joining is not uniformly an
accuracy improvement. Seed407 chr15 rises from 2,075 to 4,994 errors, chiefly
during final refinement. The [fragmentation validation](validation.md#n320-fragmentation-replay)
retains this regression and the smaller seed408 chr18 regression explicitly.

### Memory and checkpoint I/O

Intermediate refinement checkpoints store identity-bound local-row paths and
non-derived metadata instead of repeated full-chromosome arrays. Final
release products retain ordinary block arrays. Completed and partial resumes
reconstruct the original probabilities, called/inference alleles and provenance.

On the N80 chr11/chr20 integration controls, intermediate storage fell from
about 405/387 MB to 2.11/2.42 MB, with unchanged checkpoint counts of 258/327.
The later 217-chromosome matrix used about 354 MB of intermediate checkpoints
and had a maximum observed process RSS of 25.5 GiB. N320 refiner-only peak RSS
was 63.94 GiB; the fresh-feedback/assembly chr3 replay peaked at 110.90 GiB in
its parent process. These are different workloads and not aggregate node-memory
measurements.

Packed buffers, dosage/traceback tables and background/flank caches are bounded
by available memory; direct numerical fallbacks remain. Candidate concurrency
is also memory-bounded. Sampled warm N320 runs averaged about 48–54 useful
CPU-equivalents on 76-core nodes, rather than continuous full saturation.
Phase boundaries, checkpoint I/O, memory traffic and synchronization remain.

Detailed benchmark sources and artifacts stay in ignored work storage:
`.work/founder_25s_20260916_EO7tojvt/`,
`.work/n320_refiner_20260917_g3ece5Py/`, and
`.work/n320_fragment_retry_20260917_UTISglBg/`. Progressive assembly results are
in `.work/progressive_refinement_20260917_V2eVeIBe/`.
Earlier optimization-stage timings are archived locally rather than repeated
here as competing descriptions of the current implementation.

## CPU allocation and checkpoint reuse

`core/parallel.py` owns the shared process/thread budget, forkserver pools,
shared arrays and dynamic Numba allocation. Discovery reuses workers across
chromosomes in all three workflows. Active workers share the full caller's CPU
affinity; freed cores become available at the next numerical phase boundary.
A running Numba kernel cannot acquire additional threads mid-call.

Painting, genome-wide pedigree inference and family refinement use their
supplied Numba budgets with other numerical libraries limited to one thread.
Compression/decompression and raw-GL tile work are also bounded by the caller's
budget. Memory bandwidth, memory-limited worker counts, I/O and straggler tails
can limit useful activity; a nominal thread count is not a utilization measure.

Atomic stage/chromosome checkpoints retain scientific identities. A lossless
`00_genotype_evidence` cache holds GLs, positions and observed masks so T10/T11
can avoid repeatedly decoding rich simulation/discovery payloads. Original
inputs remain available; uncached runs use the original reader. This adds disk
space and one cache write, not lossy evidence or relaxed input validation.

| Bounded raw-evidence work, 112 CPUs | Before | Updated |
| --- | ---: | ---: |
| Complete GL construction, chr16 | 11.93 s | 0.81 s |
| Complete GL construction, chr3 | 48.99 s | 3.02 s |
| Repeated raw-evidence read, chr16 median | 2.10 s | 0.60 s |
| Repeated raw-evidence read, chr3 median | 8.41 s | 2.34 s |

GL timers include validation/construction, not loading the read checkpoint.
Reader timings are repeated shared-filesystem reads, not guaranteed cold-disk
measurements. On chr3, GL-construction peak RSS fell from 46.96 to 26.48 GiB and
reader RSS from 38.82 to 11.52 GiB. Compact chr16/chr3 caches occupy approximately
0.77/3.04 GiB and cost approximately 1.76/6.86 s to write.

## Stage 1: 200-SNP discovery

The canonical reversible-cavity search retains its likelihoods, three mean-field
starts, convergence, wildcard interpretation, ties and release rules.

- Binary frontiers share identical fixed-K fitting trajectories, including
  observation masks. Wholly unobserved samples become WW before usage ordering
  and founder updates.
- Equal assigned-pair/read cases share cavity calculations, predictive
  logarithms and entropy terms without rounding likelihoods or losing sample
  multiplicity. Site tiles target 8 MiB.
- Ordered founder-row/pair emissions are shared within frontiers using bounded
  scratch. Pattern tables retain ascending-set-bit sums and contiguous access.
- Exact gauge cuts decompose positive-edge connected components and solve
  bipartite components by coloring. Non-bipartite components retain exhaustive
  cuts; the high-K heuristic and production exact-search cap are unchanged.

The default mean-field cavity predictor is not itself exhaustive enumeration.
Exact gauge search can still be exponential in non-bipartite component size.
Clustering retains its jointly observed, bit-packed distance calculation and
can require quadratic storage in the residual cloud size.

| Separate warm comparisons | Before | Updated |
| --- | ---: | ---: |
| Mask-aware fitting/cavity reuse, 1,571 blocks, 76 CPUs | 99.84 s | 90.15 s |
| Further exact reuse, same batch size, first 76-CPU node | 89.74 s | 83.05 s |
| Further exact reuse, second 76-CPU node | 88.67 s | 82.09 s |
| Shared frontier emissions, 1,571 blocks, 112 CPUs | 48.95 s | 38.51 s |
| Graph simplification, all 1,647 chr1 blocks, 112 CPUs | 40.82 s | 40.31 s |

These exclude first-use compilation/pool startup. The graph simplification is
effectively neutral on the tested low-K chromosome; a roughly 22-fold bipartite
K=16 kernel improvement used a raised exact cap only in that fixture.
All 1,647 block outputs and targeted 2× dropout/missing-tract controls retained
their scientific fields. No new full-genome discovery timing is implied.

## Founder completion before L1

Let D=K(K+1)/2 be unordered diplotypes, U_l unresolved founder alleles at site l,
and C_l=2^U_l allowed allele configurations. Completion retains the shared
latent-allele model, robustness parameters, release gates and cap U_l<=6.

For each configuration update, dosage sufficient statistics accumulate sample
log evidence once. Dosage terms are quadratic polynomials in the unknown binary
alleles; known alleles are absorbed into constants/linear terms. Successive
binary table expansion evaluates all assignments. Positive-only marginal
folding supplies unary/pair dosages for the state update.

The dominant per-iteration work is
O(N L D + L D + sum_l C_l), with O(K sum_l C_l) configuration storage.
Identical suffixes of the existing five starts are shared while preserving
their initial ELBOs, first-increment checks and diagnostics. These starts do
not constitute five independent searches after their identical first update.

Whole-bin cavity completion reuses outside-bin scores, counts and separations.
Its core likelihood table improves from O(N K² 2^U) to O(N K² + N 2^U).
The public assignment table and exact carrier-sensitivity diagnostics still
retain O(N U 2^U) work. Near-cancellation polynomial values use direct positive
mixtures. There is no posterior truncation or increased enumeration cap.

| Complete preprocessing, fixed chr16 blocks, 112 CPUs | Before | Updated |
| --- | ---: | ---: |
| Seed400, 1,828 blocks | 31.08 s | 27.03 s |
| Seed401, 1,828 blocks | 30.72 s | 25.99 s |
| Seed402, 1,828 blocks | 30.75 s | 25.88 s |

All discrete fields matched across 5,484 blocks; maximum float-array differences
were 1.01e-12. A synthetic exchangeable-founder case amplified internal state
roundoff to 7.96e-6, but starts, iterations, released calls and held-out profiles
were unchanged. This is documented numerical variation, not bitwise identity.

## L1–L4 assembly

Both assembly models use the same linker and at most 20 fitting iterations.
The default bounded search uses 16 full proposal scores per category. With
that search, dense learned transitions are cubic in K; the optional
sparse-plus-background model is near quadratic. `--assembly-search broad`
retains the optimized broader full-refit search, at additional cost; the
near-quadratic whole-assembly bound does not apply to it. Search quality
and model restrictions are described in [founder scaling](founder_scaling.md).

Model-preserving optimizations include range-guarded dense BLAS contractions,
streamed scores, shared candidate prefixes, native stable top-q ranking,
streamed candidate/mate reductions, and reuse of an already scored panel.
Small/extreme dense cases retain stable log-domain arithmetic. The structured
model has its own near-quadratic extreme-value path, not a cubic fallback.
Scientific search caps and full Viterbi/BIC acceptance are unchanged by these
numerical optimizations.

| Whole four-block assembly, 320 samples, 112 CPUs | Before | Updated |
| --- | ---: | ---: |
| Dense, K=64 | 10.32 s | 8.78 s |
| Dense, K=128 | 90.25 s | 42.47 s |
| Structured, K=64 | 19.95 s | 18.69 s |
| Structured, K=128 | 83.05 s | 60.40 s |

These fixed-input comparisons retained all calls, BIC values and search
decisions. They do not establish a low-K win: warm complete chr16 calls on
seeds401/402 rose from 107.56/104.53 s to 112.11/110.50 s, and chr3 from 418.98
to 428.76 s, including added downstream-evidence cache writes.
The last non-assembly optimization round changed none of the L1–L4 algorithms.

## T09 painting and T10 pedigree inference

Painting uses bounded sample batches, direct ragged emissions and efficient
release/chunk construction. T09 coverage validation uses sorted-bin searches
and difference arrays: O(C log M + M) instead of O(C M) for C chunks/M bins.

T10 fuses strict raw normalization and gathering, reuses named-founder emissions
only where both alleles are called, and specializes exactly classified star
bridge operators. Missing/BACKGROUND states retain their general likelihoods.
The child-tiled scoring cache is capped at 256 MiB; uncached scoring preserves
the same quadratic state calculation. Deterministic transmitted alleles avoid
an unnecessary mate-state sum.

Separately, T09 can retain the upper triangle of its symmetric diploid emission
table. Its aggregate budget is at most 1 GiB per chromosome, also constrained
by an explicit painting workspace limit. T10 reuses it after existing
raw-input, sample-order and configuration checks. Transitions remain ordered;
high-K, memory-limited and uncached products recompute emissions normally.

| Packed-cache comparison, 112 CPUs | Before | Updated |
| --- | ---: | ---: |
| chr16 T10 preparation | 5.06 s | 3.45 s |
| chr3 T10 preparation | 20.74 s | 18.35 s |
| chr16 painting + checkpoint write/read + preparation | 7.32 s | 6.36 s |
| chr3 painting + checkpoint write/read + preparation | 29.14 s | 27.80 s |

Net savings are 5–13% for these measured portions, after added I/O. Compressed
T09 files grow from 48.6 to 261.8 MB on chr16 and 197.3 to 541.7 MB on chr3;
retained arrays occupy 250/406 MiB.

Earlier paired painting tests on 19 seed402 chromosomes at 76 threads reduced
painting-call time from 247.87 to 74.74 s; largest chr3 fell from 48.50 to 13.12 s.
These exclude discovery, assembly, input loading and checkpoint writes.

The latest full 22-chromosome metadata-free seed401 T10 call took **15.04 min**,
including loading/preparation/output; inference itself took **9.44 min** on
112 CPUs. Peak process RSS was about 66.9 GiB and sampled CPU activity 97.9%.
It used new packed chr3/chr16 paintings and the uncached path elsewhere.
All seven result tables matched the accepted reference: 20 M0 roots, 300 exact
M2 pairs and 600 correct edges with no extras. This is a measured duration,
not a paired whole-stage speedup or an untouched-seed accuracy claim.

## T11 family refinement and phase correction

The phase-focused product begins checking final phase after 20 iterations and
requires five consecutive unchanged checks unless family inference converges
sooner. It does not release full marginal-posterior tensors. This is a validated
scientific stopping trade-off, not a guarantee of latent convergence.

Exact implementation work skips irrelevant hard-genotype factors, contracts
zero-transition phase bins, and reuses settled selector chains and an identical
immediately preceding conditional solve. Factor-produced 32-marker dirty tiles
restrict selector-block inspections; forward/backward propagation and stopping
semantics remain intact. Ordinary/branch caches are bounded near 24/4 GiB per
chromosome. Atomic work checkpoints retain messages, phase and stability history.

Recorded complete phase-focused trials took 18.10–18.71 min for seeds400–402
on 76 CPUs; a canonical seed401 wrapper run including checkpoints took 19.03 min.
Those runs used fixed earlier q4 assemblies, not newly assembled q16 inputs.
They do not guarantee a sub-20-minute runtime on every dataset.

The latest dirty-tile comparison gave chr16 27.80 ->26.30 s and paired chr3 means
76.74 ->74.55 s on 112 CPUs: modest 3–5% gains with identical final phase and
stability results. Earlier exact reuse reduced summed 22-chromosome calls from
586.55 to 542.70 s; those sums exclude external raw reads/final checkpoint writes.

## T12 conditional recombination maps

Shared-family orientation trials use indexed four-state transfer products,
lazy parent/child XOR updates, and a separate tree for the asymmetric
orientation prior. Missing/gap/component resets, trial order, conflict refresh
and acceptance remain unchanged. After O(E L) initialization, trial/update
work is O(log J) per incident edge, with J segment leaves.

The indexed route requires a positive artifact process and strictly increasing
genetic coordinates. Disabled artifact processes or flat map stretches retain
stable streaming. A 100-Mb zero-map case exposed non-finite unguarded products;
the guarded public route matches streaming. No truncated local likelihood or
changed recombination prior is substituted.

All 22 seed401 final-phase maps passed paired comparisons. Summed paired mean
chromosome times fell from 176.59 to 146.43 s: 17.1% less wall time and 20.1%
less CPU work. Up to four workers shared 112 CPUs. These sums are not a single
112-thread whole-stage wall time. Orientation moves, called crossover intervals
and coverage matched; floating map values matched rtol=2e-9, atol=2e-8.

## Changes deliberately not adopted

- Five-bit split pattern tables: 16-fold table-memory reduction but about 7%
  slower warm discovery; some search diagnostics/assignments changed.
- Full unpacked T09 emission caching: extra I/O erased most chr3 benefit.
- Simple optimistic M2 bounds: eliminated 0 of 60,800 candidate trios; pruning
  MAP losers alone would not preserve integrated state/bootstrapping results.
- Persistent VCF handles/coarse queries and packed assembly traceback: no
  useful measured improvement on their controlled workloads.
- Approximate cavity posterior truncation, sparse-neighbour replacement of
  missing-data clustering, or weaker release thresholds: not promoted.

Local detailed measurements remain under ignored `.work/` result directories.
Loose development source and the full pre-cleanup documentation are recoverable
from `work/history/precommit_20260912_ONuv9Z/development_sources.tar.gz`; its
manifest identifies original paths. Neither that archive nor checkpoints are
part of the public package. The current scientific evidence is summarized in
[validation](validation.md), not inferred from benchmark speed alone.
