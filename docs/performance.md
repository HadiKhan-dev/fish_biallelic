# Performance and resource use

The supported pipeline uses missing-aware block discovery, dense L1–L4 assembly
with bounded panel search, ragged painting and pedigree inference, phase-focused
family refinement, and conditional recombination maps. The structured assembly
option changes the transition model; numerical reuse does not.
See [assembly choices](founder_scaling.md) and [scientific validation](validation.md).

Timings below are controlled component comparisons or explicitly labeled
recorded workflow durations. They use different inputs, CPU counts and cache
states: do not multiply their speedups or add them into a fresh end-to-end
runtime. First-use compilation, shared-filesystem I/O and chromosome complexity
matter. All simulations cited here are development data, not real-data truth.

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

Both assembly models use the same linker, at most 20 fitting iterations, and
16 full proposal scores per category. Dense learned transitions are cubic in K;
the optional sparse-plus-background model is near quadratic. Search quality
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
