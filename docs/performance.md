# Performance and resource use

## Discovery measurements for 2.6

Discovery keeps canonical birth-proposal keys and descriptors until fitting needs panel arrays,
and runs the fractional-founder scalar certificate before building term-column views. These exact
changes preserve the search budgets, objective, fit counts and released decisions.

Each row below is one single-contig pair against 2.5, run sequentially on the same 76-core Ice Lake
node with all 76 cores and symmetric kernel warm-up. Native discovery wall time measures the block
pool; inclusive stage wall time also includes preparation and startup. These are single matched
pairs, not repeated estimates or whole-pipeline speedups.

| Workload | Blocks | Discovery wall, 2.5 to 2.6 | Change | Inclusive stage wall, 2.5 to 2.6 |
| --- | ---: | ---: | ---: | ---: |
| N=80 | 1,282 | 33.3 to 30.1 s | -9.6% | 36.13 to 32.66 s |
| N=320 | 580 | 54.4 to 47.0 s | -13.6% | 58.62 to 51.24 s |
| TroMau | 834 | 32.7 to 30.6 s | -6.4% | 47.04 to 43.97 s |

All 2,696 block records agree in non-timing fields, including fit counts, omission scopes and
nonfinite classifications, with zero measured continuous drift. A separate three-chromosome std
check preserves 4,080 discovery block records and downstream scientific summary fields, also with
zero measured continuous drift. That check has fewer than the five contigs required for Tier-B
pedigree inference and does not establish parent-linked phase coverage. Scientific outputs and
decisions, rather than byte equality, define identity.

Stage CPU time falls from 2,362.1 to 2,114.3 s at N=80, 3,888.0 to 3,304.7 s at N=320, and
2,317.8 to 2,171.3 s on TroMau. Summed discovery process RSS is 22.03 to 22.02, 27.81 to 27.79,
and 27.59 to 27.23 GiB respectively; shared pages can be counted repeatedly. Both TroMau arms
logged a cache-manifest publication warning after successful computation and cache-file publication;
the warning did not affect completion, the scientific comparison or the measured block pool.

## Local-selection measurements for 2.6

Copying kernels use 64-sample lanes, narrowed to 32 or 16 for blocks with more than 64 samples
when all resulting lanes fit the held thread budget. Each kernel temporarily caps its threads at
the useful lane count, then restores the caller's mask without releasing its CPU claim. Normalization
reuses reciprocals and combines triangular normalization with marginal accumulation. Model and search
cost orders are unchanged; the trajectory memory estimate retains the conservative 64-sample width.

One matched pair per workload ran sequentially on the same 76-core Ice Lake node, with symmetric
kernel warm-up. These local-stage timers exclude final assembly and painting:

| Workload | Local-stage wall, reference to optimized | Change |
| --- | ---: | ---: |
| K30 m1 | 373.3 to 237.1 s | -36.5% |
| TroMau chr7 | 200.8 to 179.8 s | -10.5% |
| std chr1 | 120.2 to 114.3 s | -4.9% |
| N320 chr8 | 97.5 to 99.7 s | +2.2% |

The reference includes the discovery speed changes and parameter split. N320's selection rounds
are flat; its small stage increase is in unchanged context assembly and is within timing noise.
Std's smaller gain is provisional. K30 uses about 7% more CPU for the shorter wall time. These
single pairs do not establish whole-pipeline gains. Scientific gates pass for all four contigs
and full std through truth evaluation and semantic exports; maximum local log-likelihood drift
is 2e-10, with zero measured final-product, evaluation and export drift.

## Historical 2.3 integration measurements

The release benchmark uses 112 allocated cores on an AMD EPYC 7763 node. Report
host, CPU model, affinity, wall time, CPU time and memory together. The 2.1 TroMau
reference ran on 112-core Sapphire: 2,298.5 seconds and 41.74 GiB peak process RSS.
A ratio across these machines does not isolate a software speed change. The governor
uses available node memory with normal headroom, without throttling at the historical
1.5-times baseline. Report both the whole-run peak and largest single-task footprint.

### Completed integration measurements

| Measurement | Source and hardware | Result |
| --- | --- | --- |
| TroMau cold pipeline | v23c4, EPYC 7763, 112 cores | 2,947.967 s; full acceptance task 3,011.076 s |
| TroMau sampled process-tree resources | Same run | 93,379.74 CPU-s; 62.524 GiB summed RSS peak |
| Whole family-phase stage | C1 Y, EPYC 7763, 112 cores | 1,492.322 s; post-fit separately 321.142 s |
| Family-stage worker resources | Same Y run | 110,610.43 contig CPU-s; 74.12 mean busy cores |
| Family-stage memory | Same Y run | 469.605 GiB whole-tree peak; 83.560 GiB largest task |
| Fixed-genome N80 pipeline | v23c5 (G7 and C1 X), EPYC 7763, 112 cores | 327.576 s |

TroMau's raw wall ratio to the 2.1 Sapphire baseline is 1.283. Hardware differs,
so this does not establish a matched resource gate. Four observed exit-counter races
qualify its sampled CPU total. Summed RSS can count shared pages repeatedly; it is
not directly comparable to a historical peak-process value. Later G7/C1/G8 biological
identity evidence carries results, not the v23c4 timing.

The whole family-phase target is about **1,200 seconds at N=320 on 112 cores**,
originally Sapphire. Y's combined family stage and post-fit is 1,813.464 seconds;
its ideal 112-core floor from contig CPU is 987.59 seconds. The later completed AB measurement is 1,570.299 s for family phase plus 345.638 s
post-fit on EPYC, with 110,665.76 contig CPU-s and 343.2-GiB sampled peak. It did
not meet the target; fewer simultaneous contig workers made it about 5% slower than Y. Eight passing std/stress identity controls
do not establish speed. The N-series uses C1 X, so neither Y nor AB timings may be
substituted into that series. Full final truth and invariance coverage is incomplete.

### Certified exclusion shortcut

G8's completed-window certificate preserves the original genome-average exclusion
threshold, with exact completion for unproved cases. On TroMau it certifies 273 of
704 chromosome-dyads, completes 431 exactly and avoids 26.224% of pair-site visits.
Measured kernel CPU falls from 20.955 to 19.741 seconds (5.79%) on the 112-core
EPYC 7763 worker. This modest kernel saving is not a whole-pedigree speedup or a
subquadratic complexity claim. Two diagnostic tables expose certified intervals;
biological releases are identical under the approved report contract. Four-dataset
checks cover actual G8 TroMau/std artifacts and zero-target stress/sim carry-over,
plus a bounded default-handler, serialization and checkpoint-reuse fixture.

### Fixed-genome scaling design and interpretation

The prepared series uses five cropped template chromosomes (chr8, chr10, chr13,
chr19 and chr20), six founder haplotypes, depth five, three observed generations and
repeated family units of 2/4/4 samples, at N=80/160/320/640. Each chromosome retains
a fixed 15,200,363-bp prefix span from its first marker. The total is 652,126 markers,
equal to the former two-chromosome design. Original probabilities and coordinates
are preserved. Exact counts, positions and template hashes are in the benchmark
manifest. Pedigree thresholds are unchanged; observed-parent family sizes stay
fixed while family count grows.

The pedigree adapter changes only structure sampling; production meiosis and read
kernels are unchanged. The same tiny preflight precedes each timed cell outside
measured time and warms its exercised signatures; residual compilation can remain.
Measured cells start with cold inference checkpoints. One realization per N gives
descriptive slopes, not an asymptotic proof. The extractor refuses mixed hardware,
core budgets, source, design or cache regimes. N80/N320 use frozen v23c5 (G7 plus
C1 X). N80 completed in 327.576 seconds. N320 did not complete within its 2,700-second
cap; chr8 discovery alone took 1,564.7 seconds versus 14.4 seconds at N80. No
end-to-end N80/N320 exponent is available. N160/N640 remain unmeasured unless separately
reported. Later Y/AB exact swaps do not change the source of these measurements.

Native records preserve per-contig stage work times. A separate benchmark sampler
measures major-phase wall and CPU time and summed process-tree RSS every 0.25 seconds.
Concurrent local selection, assembly and painting share the reconstruction resource
row; their native individual worker times remain separate. Summed worker seconds
are not elapsed wall time. Nested pedigree phase rows must not be added to the
inclusive stage total.

The target outside pedigree remains about **N^1.2**. Pedigree's restated target is
linear expensive work with bounded plausible parents, a certified all-pairs screen
projected at at most 10% of total runtime at N=10,000 on the same genome and hardware,
and no dense quadratic default-path arrays beyond compact screen storage. Dense
downstream arrays remain, so the full pedigree target is **not met**. The later
sections distinguish kernel gains from these unresolved pipeline requirements.

Process-pool claims reserve shared capacity before admission and retain it until the
worker reduces its claim at a numerical boundary or finishes. Late workers wait when
no CPU is free. Counter fixtures, bounded four-core numerical fixtures and paired K30
scientific identity pass. Family phase uses the matching in-process claim ledger.
Stage6ab adds first-task work-based initial shares while retaining remaining work for
dynamic growth, and admits scratch before nested CPU claims. Focused admission and
first-pass tests preserve the model and CPU accounting; full-run utilization is
measured separately.

### Validation coverage at freeze

This is the historical 2.3 acceptance state, reconciled with the release changelog.
The 08:22:56+01:00 transfer snapshot had no terminal NOT_RUN records, which did not
make its queued/running cells passes. Component receipts, compositional carry-over
and fresh final-pipeline runs are distinguished below.

Std reference carry-over is established for discovery by all 11,822 saved c3 blocks on eight contigs: K1 has at most 80 samples and the largest residual pool is 184, below 256. Partner pools are subsets of the global pool. The inactive branches retain the dense code; exact clique changes and C1/G7/G8 deltas use their accepted identity evidence. This is compositional biological carry-over of the c3 reference, not a fresh final-tree run or proof for perturbed inputs.

Actm3's saved c3 residual pools reach 292, 310 and 270 on the three checked contigs,
so A256 can activate. Its c3 result does **not** carry to the final tree. No c4 actm3
reference has completed; final actm3 evidence is component-only. TroMau completed
on c4 and carries biologically through the accepted exact swaps, with G8's explicit
interval-report contract. Completion is not a truth or reversal/noise pass.

The uncompleted truth references are stress_n80, lad_K20_n80, lad_K30_n80, f3_fam,
backcross_s6202, reallike6_nf_s3, actm3 and sim_s10000. The queued c4 std reference
has not started. Both std and stress still lack their final iid12_s1, reversal and
rename_s1 controls. N320 timed out at 45 minutes, N160/N640 are unmeasured, and
AB whole-stage timing completed as reported above. TroMau reversal, the broader zero loop and final Gate W remain
uncompleted. These are evidence gaps, not demonstrated failures. Release 2.4 added evidence
identity on all 22 TroMau chromosomes and std chr1/chr2, plus TroMau chr19 downstream
identity and a three-chromosome end-to-end check; it did not rerun the simulation matrix.

Release 2.5 combines the exact residual-tail remedy, inline small residual work, guarded fractional
fixed-K reductions, discovery's CPU handoff to waiting block workers, discovery diagnostics and mode
removal, the population-half proposal family and an operational family-phase typing fix. Measured on
76-core Ice Lake nodes:

- Discovery on the five N-series chromosomes takes 235 s at N=160 and 369 s at N=320, against 7,973 s and
  10,266 s in 2.4.
- The whole N-series pipeline takes 10.6, 14.6, 22.6 and 45.6 minutes at N = 80, 160, 320 and 640, a
  scaling exponent of about 0.8. The N=80 run includes first-use kernel compilation.
- TroMau takes 67.7 minutes, against 72.4 in 2.4.

Released results equal 2.3 on TroMau (all 22 chromosomes) and on the std, stress, K20 and K30 simulations.
Successive 2.5 candidates were compared block for block and downstream. Historical gaps remain explicit, and
component checks do not substitute for full runs.

### Saved TroMau cross-version attribution

The matched 2.2/c4 saved products first differ at discovery on all 22 chromosomes:
1,028/9,979 blocks change in known latent content, calls or retained alternatives;
455 change selected K. Initial/L1/L2 selected discrete products differ in
272/376/540 blocks respectively. All 22 final internal assembly/refit chromosome
panels change. These observations locate the first changed interface, not an
exclusive cause for every downstream change.

The exports have 33,301 changed genotypes among 230,485,131 common-called sample-site
cells, plus 118,698 call/unknown disagreements among 686,590,544 total cells. The
13,942 relative phase switches compare 101,413,928 eligible consecutive heterozygous
observations within matching pairs of released phase sets. None is an accuracy/error
rate against truth. The one changed parent release (1/116 samples) changes from
unresolved to a supported two-parent state; both versions selected the same local
pair, while the state bootstrap fraction changed from 0.674 to 0.709. Pedigree modelling
also changed in 2.3, so saved matched outputs alone cannot assign that transition to
discovery. No inference was rerun for this attribution.

## Where the time goes

| stage | parallel unit | main cost |
| --- | --- | --- |
| evidence | contig (metadata scan, then direct shared-array decode in parallel windows); at most one contig ahead of discovery | site/sample QC, read-model fits; 33 bytes per retained cell, plus phase scratch |
| discovery | block, in one persistent pool and one block queue for all pending contigs | fixed-K fits and the reversible K search per block |
| local selection | block (path fits), component (context refits), inside a contig lane | path-model fits and the candidate bank; two context assemblies |
| assembly | joined batch (hierarchy), component (founder refinement), inside a contig lane | linker EM and certified panel searches; converged refinement fits and count decisions |
| painting | sample block, inside a contig lane | binned emissions and the diploid HMM |
| pedigree | chromosome (preparation, scoring, exclusion); genome-wide decisions are a barrier | pair scoring on the fixed candidate panel; bootstrap and leave-one-chromosome-out resampling |
| founder refit, family phase, recombination | chromosome | coordinate sweeps; family message passing and polishing; per-meiosis decoding |

Evidence-only allocation measurements for 2.4 (cpu-q-307, 18-core ceiling, base23 versus E2;
site/sample filters unchanged) are below. Values are peak parent RSS in decimal GB and include
allocator/JIT overhead. They are per-phase peaks, not additive. The largest contig is TroMau chr3,
N=116, 749,986 input SNPs, 127,172 kept, one flagged sample.

| Evidence measurement | base23 | E2 |
| --- | ---: | ---: |
| chr3 source decode/canonical ordering | 3.077 GB | 1.532 GB |
| chr3 site filter including QC | 4.952 GB | 4.333 GB |
| chr3 read model | 1.796 GB | 0.492 GB |
| chr3 likelihood emission | 2.185 GB | 0.848 GB |
| chr3 record write | 1.713 GB | 1.074 GB |
| chr3 founder-view construction | 2.031 GB | 0.923 GB |
| N=10,000, M=2,048 generated-BCF decoder, added bytes/input cell | 32.66 | 8.25 |
| N=10,000, M=2,048 synthetic GL emission | 1.234 GB | 0.834 GB |
| N=10,000, M=2,048 synthetic founder-view construction | 1.544 GB | 0.891 GB |

The filter's lower RSS reflects fewer retained decoder allocations, not an E2 filter change.
At N=10,000 and chr3's present retained count, current evidence plus its founder view is
43.24 rather than 83.93 GB; with three times as many retained sites, 129.72 rather than
251.80 GB. The next contig, site-filter scratch and discovery workers are additional. E2 alone
still projects about 403 GB for current evidence/view plus the next base23 filter at today's
retained count: E1's filter changes must be included before claiming a complete large-N remedy.

Six-worker isolated chr19 decode time was 1.886/1.899 seconds (base/E2); the extra metadata scan
costs worker CPU while eliminating whole-count result transfers. A three-worker decode alongside
three real discovery workers measured 1.957/2.128 seconds and did not reproduce the production
3–4x slowdown. These bounded warm-cache observations do not establish full-node overlap speed.
Both pools still receive independent full driver budgets. All required bounded evidence arrays
and report values were bit-identical apart from timings; detailed allocation formulas are in
[Evidence implementation and cost](methods/evidence.md#implementation-and-cost).

The [methods](methods/README.md) give each stage's complexity in the founder count K, the samples N and the sites L.
Checkpoint I/O, process start-up, ordered beam and panel updates, and the dependencies between stages can leave
CPUs idle at times; the design keeps them busy where independent work exists, but sustained full utilisation is
not claimed.

## CPUs

The run's ceiling is `[execution] cores` (default: the CPU affinity). It is divided as leases of one tree (run ->
contig lane -> pool -> worker), never exceeded by the product of processes and threads:

- **Contig lanes.** After discovery, contigs are reconstructed in concurrent lanes: one lane per
  `execution.contig_lanes.CONTIG_LANE_CORES = 16` cores (narrow lanes use fewer core-seconds per contig, because
  path selection scales fully but hierarchy tails, refinement tails and serial checkpoint phases do not), at most one
  per pending contig, limited by memory. Lanes start with the longest contigs, one each; a lane that finishes a
  contig takes the next from one shared queue, longest first. Leases start in proportion to each lane's work and grow
  as finished lanes release theirs: at contig starts, at the phase starts the stages mark, and inside pool phases.
- **Pools.** A pool task claims `floor(total / active)` threads plus a share of the remainder at its numerical
  boundaries; during a batch's start-up every worker has one thread until the batch's workers have started.
  Freed threads move to surviving tasks at their next boundary; a running Numba kernel never changes its team. Discovery
  enables an additional handoff at the owner's existing apply/claim boundary: when
  memory-admitted block workers wait for their first CPU, a borrower caps its team at
  max(1, floor(total / (active + waiting))). The same call resizes its Numba team; pure
  budget queries retain running claims. Growth resumes as waiters drain. Other stages
  keep the default policy, and memory admission/reservation rules are unchanged.
- **Shared arrays** are copied into shared memory once per pool, with at most
  `execution.shared_arrays.COPY_THREADS = 16` threads (the copy saturates memory bandwidth at about 16 threads).
- **Serial mode** runs one contig at a time with every core: less throughput on a large node, smaller units of
  durable progress ([usage.md](usage.md#short-allocations)).

`[execution] profile = true` prints per-phase timings and the thread-allocation trace of the leases. Every run
record holds exclusive stage timings (stages timed inside another stage on the same thread are subtracted, helper
threads are recorded under their own name) and per-contig summaries.

## Memory

One memory governor per run (`execution/memory.py`) admits work using estimated memory ownership and current
available memory. Its limit is a planning limit: a measured peak can exceed an estimate, and the progress rules
below can admit work over the limit.

- **Limit.** At startup, available memory is the minimum of node availability, enclosing cgroup limits minus their
  working sets, and the job allowance. Headroom is `HEADROOM_FRACTION = 0.1` (at least 4 GiB, at most half);
  `[execution] memory_limit` can lower the resulting limit.
- **Task workspaces.** Pools, chromosome executors and concurrent fits reserve estimated active workspace before
  dispatch, releasing it after the result. Nested work borrows an enclosing task's envelope where the existing
  path declares one. Worker and caller residence is accounted separately. New pool-owned shared arrays are held
  once for the pool lifetime; attaching an existing `SharedArrayRef` adds no new allocation.
- **Worker residence.** Discovery, local selection, context refit, hierarchy and founder refinement reserve
  measured private-memory allowances from pool creation through worker join, including idle tails. Worker-count
  planning includes each allowance once. These terms cover imports, loaded kernels, compiler/runtime state and
  retained heap; they are neither measurements of machine code alone nor universal upper bounds. Task workspace
  and shared-array terms remain separate. Existing trimming and recycling policies are unchanged.
- **Contig arrays.** For N samples, L sites and D decoded discovery-record bytes, the estimate is
  `D + 63*N*L + 200*(ceil(D/13) + SEARCH_GROWTH*L)`. The 63-byte term covers raw and neutral float64 genotype
  likelihoods, float32 scoring likelihoods and three observation masks. The extra founder-site allowance covers
  overlapping selection generations, context proposals and checkpoint copies. Named array subtotals reach
  138 bytes per founder-site in selection and 167 in the second context; 200 is a planning allowance, not an exact
  bound on every Python object, memo or decoded backing buffer. `SEARCH_GROWTH = 8` estimates growth and does not
  cap founder counts. Pool-owned evidence copies remain additional reservations.
- **Caller residence.** Each concurrent lane also holds a process allowance until its executor joins. A driver
  executing reconstruction directly holds its allowance across later contigs and genome-wide stages, through
  cache publication. Both serial and planned one-lane routes additionally reserve each contig's arrays. The driver
  allowance is a resident token, so retained pages do not masquerade as active computation during admission waits.
- **Refinement guard scratch.** Whole-component traceback/event storage is reserved only during guarded
  acceptance, with the unchanged full-width and maximum-thread bound. Directional search messages, proposal
  caches and sparse carried history remain reserved across proposal waits; worker/lane residence and live
  availability checks are unchanged. Moving the scratch hold changes admission timing, not the guard or
  scientific decisions.
- **Admission.** A request starts when its bytes fit alongside existing reservations and current availability;
  otherwise it waits. Nested waiting work takes precedence, then arrival order. A smaller request passes a larger
  waiting one only if both fit. At depth d, the ledger ceiling is the limit minus
  `NESTED_HEADROOM_FRACTION = 0.2 / 2**(d-1)` of it. Lane residence plus its input-array plan can use up to
  `LANE_RESIDENT_SHARE = 0.5` of the limit. These rules pace memory use without changing CPU ceilings or scientific
  decisions.
- **Progress.** A task larger than the limit can run alone when nothing else holds memory. If all holders are
  waiting, the first waiter can be admitted over the limit to make progress. Both cases are logged. The governor
  does not terminate processes.
- **Measurements.** Run records report reservations, waits, forced/run-alone admissions, available-memory fall and
  incremental task RSS high-water marks. Available-memory fall is a live-use proxy, not process PSS; incremental
  task RSS includes first-use work and can include nested/shared mappings. Scientific phase boundaries release
  arrays and pools, but runtime and allocator pages can remain resident after `gc.collect` and `malloc_trim`.
  Retained memory can overlap subsequent phases.
- **Shared memory of ended runs.** Segments are named for their run. Before measuring availability, a new run
  cleans this user's segments and ledgers for runs with no live holders.

Pacing changes when work runs. The estimates do not alter likelihoods, convergence, founder counts, release rules,
phase, pedigree or recombination models.

### Calibration in 2.7

The allowances below come from the six completed W1 cases and targeted full-node and bounded measurements.
Observations use decimal GB/MB unless GiB/MiB is written. These terms are empirical planning envelopes.

| Owner | Allowance | Observed basis and limitation |
| --- | ---: | --- |
| Discovery worker | 512 MiB | 486 MB private memory after native trim; mixed runtime and retained heap |
| Discovery active workspace base | 1280 MiB | Historical TroMau ordinary-task incremental RSS exceeded its old plan by 752 MB; the added 768 MiB leaves 53 MB margin |
| Local-selection worker | 768 MiB | 674 MB private memory after native trim in a bounded first-use K29 fixture |
| Local-selection active workspace base | 2048 MiB | Historical TroMau first-task incremental RSS 3.023 GB; named workspace 356 MB and a 106 MB bounded starting-private projection leave 180 MB margin with residence; historical starting-private memory is unavailable |
| Context-refit worker | 512 MiB | Replaces the old 256 MiB per-task startup allowance; historical first-task incremental RSS reached 362 MB, current warm private endpoint 116 MB |
| Hierarchy worker | 2048 MiB | Sequential K30 normal post-trim private maximum 1.994 GB includes current task state; captured settled-idle maximum 1.480 GB does not establish that the larger endpoint disappears after return |
| Refinement worker | 3584 MiB | Sequential K30 settled-idle private maximum 3.551 GB and normal post-trim maximum 3.692 GB; current task state remains live at normal trim |
| Lane process | 17664 MiB (17.25 GiB) | Long K30 native post-painting private endpoint 18.401 GB at a maximum 76-core lease; 121 MB margin. The last lane joined before a settled-idle sample; earlier lanes retained less |
| Sequential driver | `max(1536 MiB, 512 MiB + 272 MiB * cores)` | Five-contig post-trim private endpoints rose 15.366→19.395→20.285→21.256→21.570 GB; cache-publication endpoint 21.643 GB. Calibrated at 76 cores with 570 MB margin; core-count scaling and longer sequential lists are unvalidated, and no plateau was established |

The long resource fixture has 80 samples, 30 synthetic founder rows per template block, and 322,664 sites across
three contigs spanning about 137 Mb on a full 76-CPU SL4 node. Only coordinates come from TroMau; alleles are
resampled K30 template blocks with fixed row identities. Repeated blocks make this a resource fixture, not a
representative genome model. Three contigs do not meet the unchanged five-chromosome Tier-B eligibility criterion,
so parent-conditioned family-phase memory is not stressed.

The long run completed all stages and three exports in 12,510.105 s (12,552.568 s including observation).
Released founder counts were 30, 30 and 29 across 41.031, 52.304 and 43.842 Mb components; all three visited K30
in final refinement. Whole-tree sampled PSS peaked at 70.223 GiB and forkserver-worker PSS at 69.461 GiB.
Native governor reservations peaked at 391.619 GiB, with no waits, forced admissions or run-alone events.
There were 2,520 samples at a median five-second interval. CPU-delta measurements give a lower bound of 63.693
occupied cores on average; a 60-second window reached 75.699 of the 76 available cores.

The corrected terms preserve all eight recorded lane ceilings, and 624 logged local/hierarchy groups fit their
stated outer ownership. A simultaneous late-L3 graph would, however, exceed the full planning limit by 2.485 GB
when given all three worst-case lane allowances. That is a pacing risk, not an admission replay or a runtime
prediction. The estimates retain the CPU ceiling and existing admission policy; integrated measurements must
report any resulting waits and runtime change.

The separate sequential K30 run completed five 8000-site contigs with reconstructed K28–30 in 2221.121 s on
76 CPUs. Whole-tree sampled PSS peaked at 40.607 GiB; native governor reservations peaked at 46.631 GiB, with no
waits, forced admissions or run-alone events. End-of-run private retention exceeded 20 GiB despite native
collection and trimming, motivating a driver owner that persists into later stages. The 32.059 GiB individual
active driver PSS peak is not used as the retained baseline.

PSS/USS samples cover verified descendants about every 5 seconds. A sweep reads processes sequentially, and the
sampled peak is a lower bound. Summed RSS counts shared mappings repeatedly. Concurrent phase maxima overlap and
must not be added. The original long observer captured all descendant PSS but lacked some nested task/trim hooks;
worker roles are recoverable from ancestry, missing boundary observations are not. Corrected bounded and
sequential probes provide direct task and native-trim measurements. Historical six-run comparisons have 30-second
RSS samples rather than PSS. Their outer family-phase ratios and VCF window RSS do not establish an unreserved
private-memory deficit: nested reservations and separately owned shared output pages also contribute, and their
simultaneous historical ledger was not recorded.

Warm repeated-task fixtures preserve released scientific fields with unchanged cache contents. Their retained
memory is far below some first-use and full-run endpoints, so a warm-only constant would miss supported cold paths.
Native trimming returns some pages, but finite repeats do not prove a universal plateau or identify allocator
ownership. No recycling-policy change is made.

## Compiled kernels

Every `@njit(cache=True)` kernel compiles into a cache directory per code version: the key covers the whole
package source and its location, the Python, NumPy, Numba and llvmlite versions, the code-generation settings and
the CPU model. Numba's own cache checks only each kernel's file timestamp. Workers of a run share the directory.
Compiled kernels persist between runs in `~/.cache/haplotype-reconstruction/numba`, or beneath `$XDG_CACHE_HOME`
when set. The `execution.kernel_cache.KEEP_KEYS = 6` most recently used code versions are kept.
`NUMBA_CACHE_DIR` sets the base of the run's per-version directories
([configuration.md](configuration.md#environment)). Normal runs need no cache configuration.

A run first loads the stored compiled snapshot for its code key, when one exists. Before scientific pools
start, it requests the union of the package's release signatures and the persistent store's learned
`manifest.pkl` signatures, excluding overloads already cached for the current code-generation target. With
compatible dependencies, an empty persistent store still warms the release signatures. A source or CPU change
gets its own code key; the release list requests compilation on the current machine and ships no compiled
machine code.

Warm-up uses the run's CPU budget, with one task per function and its requested signatures, longest expected
compile time first. Timing hints affect scheduling only. Different functions can compile shared callees, so
some compilation can still be repeated across warm-up workers. The cached-argument signature check introduced
in 2.6 remains: an entry whose compiled argument types differ from the requested overload is a cache miss and
is compiled again. Release warm-up does not bypass the code key, current-target check or signature guard.

When at least one manifest overload is missing at startup, the parallel pool closes before a fresh single
worker checks the full release/learned union. It loads each listed signature through the same argument guard
and compiles misses without competing warm-up writers. This repairs entries lost or mismatched when parallel
tasks save overloads of shared callees. The check makes one pass over the finite manifest; it does not repeat
until every entry succeeds or introduce another cache format. Unsupported signatures retain lazy fallback.
If the initial cache index shows no missing requests, this serial check does not run. Index presence alone
is not proof that every payload can load; the first-use signature guard and lazy compilation still apply.

A run publishes its compiled cache and observed signatures when it gains kernels, including after a scientific
failure. New signatures absent from the release list compile on first use and can enter the learned manifest.
Unavailable functions or signatures are skipped during warm-up; unreadable manifests and store failures are
logged and retain compilation on first use. The cache is disposable compilation state, not scientific
checkpoints. Changing when compilation occurs does not change the numerical kernels, model, thresholds or
scientific outputs; release validation compares scientific decisions and continuous drift, not compiled bytes.

### Regenerating release signatures

After integrating the final candidate, collect learned `manifest.pkl` files from its completed reference runs
and regenerate the release list with that candidate on `PYTHONPATH`:

```bash
PYTHONPATH="$CANDIDATE" python -m haplotype_reconstruction.execution.kernel_manifest \
  reference-a/manifest.pkl reference-b/manifest.pkl
```

Here `CANDIDATE` is the final source-tree directory. The default output is that imported candidate's
`haplotype_reconstruction/execution/kernel_manifest.json`; `--output` selects another destination. The tool
unions the reference signatures, resolves their current importable package dispatchers, skips removed functions
and writes entries in sorted order. Regenerate after later source changes so new kernels from those references
can participate; copying an earlier release list does not discover their signatures.

The shipped file is text JSON. Each entry contains a readable argument-type description and a base85-encoded
pickle of the exact Numba signature. The pickle preserves literals, omitted defaults, array layout/read-only
flags, named tuples and typed containers; the readable description is not a type parser. The file records
the generator's Numba version. Signatures use Numba's cache serializer, including the dynamic ctypes
function-pointer types used by site-quality kernels. Warm-up transports those types as serialized bytes,
and the learned manifest uses the same serializer; standard pickle readers remain compatible. No
function pointer address or compiled machine code is shipped. Type portability still
depends on compatible Python/Numba types and importable names. Signatures that cannot be decoded or compiled in
the current environment are skipped and keep the usual lazy-compilation fallback.

Reference coverage is finite. A new input dtype, configuration, kernel or dependency version can require an
unseen signature; a release manifest cannot guarantee that all future scientific workers avoid compilation.
Validate warm-up with a fresh persistent store and run-local cache, and report requested/skipped overloads and
any signatures first observed after warm-up alongside the scientific comparison. The learned/release union
extends coverage without changing inference or treating missing coverage as permission to load another overload.

A fresh-cache, five-contig N80 validation on 76 Xeon 8368Q CPUs completed in 508.917 s of raw run-record wall
time, including 0.114 s of measured pre-science cache inventory. Warm-up took 63.383 s, including the 4.8 s
serial check. All 749 shipped signatures were present after warm-up and after science, with no runtime additions.
The complete c0 scientific comparison covered 7,887,042 records per side with zero differences or observed drift;
all 3,263 discovery blocks were unchanged. With matching science configuration and 76 Xeon 8368Q CPUs, the c0
reference took 523.916 s and W3's warm run took 447.168 s: this candidate was 61.749 s slower than W3's warm run.
These are single attempts on separate nodes; W3 uses different scientific source and the runs have different
manifest coverage, so the timings do not isolate the effect of this cache change.

The interim TroMau integration completed inference but exposed a cache-publication failure: standard pickle
could not serialize a dynamic `CFuncPtr` type. Its post-run inventory contained three site-quality overloads
absent from the 758-signature warm-up inventory. The release serializer now matches Numba's own serializer;
regeneration includes these observed types. A fresh two-worker test compiles both affected special-function
kernels during warm-up, loads them through the unchanged signature guard in fresh workers without cache
misses, and reproduces the SciPy reference values. Publication and reload pass. This focused result does
not replace the final merged matrix's cold-cache coverage check.

## Checkpoint I/O

Records are protocol-5 pickles whose array buffers are written out of band in Blosc2-ZSTD chunks
(`store.codec.CLEVEL = 5`; assembly records and work states, which are superseded often, use
`store.codec.FAST_CLEVEL = 1`). Large payloads overlap chunk reads, checksums and (de)compression on a rolling
pipeline with at most 1 GiB in flight; decoding is zero-copy into one aligned allocation, and readers decode only
the entries they ask for. Compression threads come from the writer's lease. Contig files grow append-only and are
never compacted (superseded records are small).

## Determinism

Results do not depend on the number of cores, lanes, workers or helpers, on memory pacing, or on completion order:
parallel kernels add per-sample (or per-site) rows in a fixed order instead of team-size-dependent reductions;
pools return results in task order (or in an order their callers do not depend on); random streams are fixed
before scheduling. The only intended randomness is a simulation with `seed = "random"`, whose realised seed is
recorded. Search limits that bound memory are fixed constants, never the memory that happens to be free: founder
refinement's window programmes, for example, are solved within growing budgets reserved from the governor up to a
fixed cap (`AssemblyConfig.refinement.frontier_bytes = 137438953472`, 128 GiB), and a solve within a budget computes
exactly what a solve at the cap computes. The same configuration and inputs therefore give the same records at any
allocation size.

## Founder counts

The pipeline is designed for up to 30 founder haplotypes per block. Exact branches with exponential cost are
bounded (exact site posteriors over at most `assembly.founder_alleles.site_posterior.MAX_EXACT_BITS = 10` free
founders; exact maximum cuts up to `DiscoveryConfig.search.exact_cut_max_k = 12`), with polynomial methods above
them, and the number of search steps is bounded by reported backstops that grow linearly in K. Founder refinement's
exact window programmes are the one data-dependent term: their frontier can grow as R^W in the worst case (R rows
per block, W = 8 fine blocks or 4 coarse spans), and the fixed frontier cap bounds it, with every fallback
reported. Each methods chapter states its stage's complexity, including the worst cases those backstops allow.

Measured on the package's own K-ladder simulation: every 200-site block carries K distinct founder haplotypes;
5 contigs x 8,000 real marker positions, N = 320 samples, one contig lane; CPU seconds per unit of work, the growth
exponent from K = 20 to 30 and the peak resident memory of one worker at K = 30 (tree of 17 October, before the
founder-refinement speed-ups, which leave results unchanged):

| stage (unit) | K = 12 | K = 20 | K = 30 | growth 20 -> 30 | worker peak at K = 30 |
| --- | --- | --- | --- | --- | --- |
| discovery (block) | 4.3 | 12.1 | 33.4 | K^2.5 | 0.70 GiB |
| local selection, initial round (block) | 15.1 | 32.1 | 60.8 | K^1.6 | 0.78 GiB |
| local selection, round l1 (block) | 0.4 | 3.1 | 9.2 | K^2.7 | |
| context refit (component) | 29 | 80 | 217 | K^2.5 | 0.60 GiB |
| linker and join (10-block batch) | 14.4 | 26.7 | 88.7 | K^3.0 | 1.40 GiB |
| founder refinement, level 1 (10-block component) | 31 | 113 | 445 | K^3.4 | 3.26 GiB |
| founder refinement, level 2 (40-block component) | 324 | 1,038 | 3,156 | K^2.7 | |

Every stage ran to recombination at K = 30 with no backstop hit and no frontier fallback (largest window frontier 257
entries), and discovery recovered 12.02, 20.04 and 30.01 rows per block. Whole runs took 511, 1,277 and 2,376 s.
With weakly supported founders (N = 80: about five carriers per founder per block at K = 30) the data-dependent
multipliers grow faster than K^3 from K = 20 to 30 (local selection K^4.2 to K^4.8 per block, context joins K^4.7,
founder refinement K^4.0 at level 1 and K^7.2 at level 2), and the K = 30 run took seven times the K = 20 wall time;
the per-step kernels themselves stay within O(K^3).

## Reference timings

Release 2.0 against release 1.0 (commit 92b3daf), whole genomes, each run alone on a dedicated 112-CPU Sapphire
Rapids node (Xeon Platinum 8480+) with a fresh kernel store, in the same software environment; largest process =
peak resident memory of one process, node peak = the node's used memory (total minus available, sampled every 30 s):

| run | release 1.0 wall | release 2.0 wall | ratio | largest process, 1.0 -> 2.0 | node peak, 1.0 -> 2.0 |
| --- | --- | --- | --- | --- | --- |
| simulation seed 10000 (N = 320, 22 contigs) | 2:08:46 | 2:03:41 | 0.961 | 27.4 -> 20.0 GiB | 301.7 -> 259.1 GiB |
| simulation seed 10001 | 2:13:49 | 2:09:56 | 0.971 | 28.5 -> 19.9 GiB | 312.6 -> 257.7 GiB |
| backcross simulation (N = 80, 22 contigs) | 40:46 | 38:53 | 0.954 | 8.9 -> 5.9 GiB | 169.9 -> 136.3 GiB |
| TroMau (116 samples, 22 contigs) | 41:00 | 41:10 | 1.004 | 6.4 -> 5.3 GiB | 114.2 -> 96.3 GiB |

The two TroMau runs ran at the same time on two identical nodes; 41:00 against 41:10 is equal within run-to-run
noise. The simulation rows were measured with the accepted tree, to which the release adds only changes with
identical results and the export stage (26 s on TroMau). Evaluating a simulation against its truth takes 0:30-0:38
instead of 1:16-1:37.

Exclusive main-thread stage times in seconds, from the run records of the seed-10000 simulation (accepted tree, no
export stage) and of TroMau (release tree). The evidence stage runs on a helper thread while discovery works, so its
time overlaps discovery's and is not part of the sum:

| stage | simulation seed 10000 | TroMau |
| --- | --- | --- |
| simulating the inputs | 72 | - |
| evidence (helper thread) | 303 | 448 |
| discovery | 668 | 472 |
| contig lanes: local selection, assembly, painting | 5,765 | 1,813 |
| pedigree | 553 | 76 |
| founder refit | 12 | 3 |
| family phase | 285 | 40 |
| recombination | 32 | 14 |
| export | - | 26 |
| outside any stage | 30 | 24 |
| run wall time | 7,417 | 2,468 |

The contig lanes take about three quarters of the wall time in both runs, discovery a tenth of the simulation's and
a fifth of TroMau's.

Concurrent runs belong on separate allocations with separate output roots. On a shared node, the governor's limit
starts from the memory available when the run starts; set `[execution] memory_limit` to leave room for other work.

## Earlier integration measurements (v23c2/v23c3)

The earlier v23c2 integration combined stage6v family phase, the G4 genome-wide pedigree model and exact-only hotspot3 discovery. G4's cached-score replays took about25/27/51/32seconds for std/stress/sim/TroMau on112 EPYC7763 cores; these are not full-stage timings. Raw M1 remains quadratic and downstream floor rows remain dense. The retained M2 panel has no arbitrary cap, so its cost depends on the data-supported parent counts.

Hotspot3 counts exact residual clique size classes before materialization; dense distances remain quadratic. Published warm residual slopes1.077(wall)/1.078(CPU) and the28.25x N640 speedup were measured with the A256 pilot enabled. They are not exponents for that earlier exact-only configuration. Default-on landmarks entered v23c4, as described below.

Stage6v isolated chr3/chr12 fits took733.290/407.284seconds on112 EPYC7763 cores; the former U runs shared a concurrent whole-stage budget, so these are not matched speedup ratios. Whole-stage1200seconds and N^1.2 acceptance remain open. Later exact changes require explicit identity checks before earlier validation is carried forward. Current release-panel tasks record frozen source manifests, hardware, native stage timings, sampled process-tree CPU/RSS and per-process high-water statistics. Sampled RSS may count shared pages more than once.

### Integrated W and calibration aggregation measurements

Stage6w preserves stage6v released fields in the supplied identity gates. Its transition lookup, allocation-free cavity accumulation and independent-site parallelism are integrated; these gates are not a whole-stage 1,200-second result. G5 calibration aggregates identical complete chromosome vectors with identical priors and reciprocal masks. On the paired 112-core EPYC 7763 replay, the sim_s10000 fit drops from 7.357 to 4.246 seconds (42.3%); this is predominantly serial optimizer work, not a whole-pedigree speedup. Exact M1 screening and dense layouts remain. The final measured complete-pedigree N80/N320 cohort uses v23c5; its hardware-specific wall, CPU and sampled RSS slopes are reported separately above.

## Fractional founder reductions and discovery CPU handoff

Fractional H updates accumulate costs along contiguous site rows. A conservative
floating-point interval certifies each allele decision; intervals touching a decision
boundary use the accurate expansion-sum reference. The workspace computes its term bound
once. Sample terms, founder order, usage sums, tie quantum, both schedules and stopping
rules are unchanged. The CPU handoff described above applies only to discovery and
lets admitted waiting blocks begin at existing numerical-team boundaries.

Matched native discovery timings use one sequential pair per 76-core Ice Lake node,
empty compiled caches and the same 751-overload warm-up manifest in isolated cache stores.
Both arms warm 746 applicable overloads; changed H signatures finish compiling before
measurement. Results compare the timed-flush/inline-work baseline with both exact speedups.

| Workload | Baseline wall | Guarded H + CPU handoff | Reduction | Identical block products |
| --- | --- | --- | --- | --- |
| std_n80 chr8 | 38.6 s | 33.3 s | 13.7% | 1,282/1,282 |
| N320 chr8 | 72.9 s | 53.5 s | 26.6% | 580/580 |
| TroMau chr7 | 45.0 s | 32.7 s | 27.3% | 834/834 |

All 2,696 blocks preserve scientific and non-timing diagnostic records, including raw
omission scopes, with zero observed continuous drift. Worker CPU is 2,386.9 -> 2,320.5 s,
4,578.4 -> 3,735.1 s and 2,692.7 -> 2,093.1 s respectively. Accumulated worker wait
reductions overlap and must not be interpreted as recoverable chromosome wall time.
Internal scratch-sum drift reaches 6.25e-13 in the fractional fixture; persisted results
are unchanged. This is scientific identity, not a requirement for identical floating-point bits.

A three-contig std_n80 downstream pair preserves all 4,080 discovery blocks and all six
founder/sample VCF contents. Its full wall time is 615.12 -> 630.49 s (+2.5%), with
asymmetric first-use cache treatment; this is an identity check, not evidence of an
end-to-end speedup. The controlled discovery pairs above provide the timing evidence.

## Exact residual-tail remedy in 2.5

Matched Ice Lake measurements on 76 cores show N320 chr8 discovery falling from
**1,327.6 s in 2.4 to 84.2 s**, and N160 from **888.4 to 48.6 s**. The complete
five-contig N320 N-series pipeline takes **27 minutes**. These measurements precede
the small-work and checkpoint-flush refinements and do not establish an N^1.2 exponent.

The residual-group tail entered in 2.3 and made larger simulations impractical:
sim_s10000 (N320, 22 contigs) chr1 discovery took **79.1 s in 2.2 versus 2,476.9 s
in 2.3**; sim_s10001 had not finished chr1 after 80 minutes. These are discovery
stage measurements or censored observations, not completed whole-genome timings.

The first integrated remedy also exposed a small-work regression: matched std_n80
chr8 discovery took **38.9 s in 2.4 versus 76.2 s**. Executor creation, reservations,
thread dispatch and cache copies on tiny residual calls accounted for the overhead.
Small residual work now follows an inline path, with concurrency reserved for substantial
independent work. Per-block diagnostic fsyncs added **4.6%** in a separate diagnostic
measurement. Completed blocks and diagnostics instead share scientific chunk commits:
64 blocks or the next completion after about 30 seconds, plus the final partial chunk.
The timer is checked at completion, so it does not guarantee a flush during a long wait
for another result. These execution/I/O refinements preserve scientific decisions;
their final regression and identity gates are reported separately.

The isolated inline-work remedy passes matched small-N discovery checks on 76 cores:
**std_n80 chr8 40.0 s versus 38.9 s in 2.4 (+2.83%)** and **TroMau chr7 46.7 s
versus 46.3 s (+0.86%)**. Runs used the same host/configs sequentially after low-load
checks, with separate per-tree kernel caches; discovery stage timers exclude startup.
All 1,282 N80 and 834 TroMau blocks retain the compared scientific fields, with zero
observed continuous drift. These gates isolate the execution remedy; final combined
checkpoint-flush and downstream checks remain separate.

The refined exact remedy completes N320 chr8 in **73.75 s on 76 Ice Lake cores**,
with 4,663.76 task CPU-s and 29.31 GiB sampled summed process RSS. All 580 saved block
products retain scientific decisions with zero observed drift; diagnostic context
suffixes alone differ. The 76.58 s earlier component run used a different Ice Lake node,
so that pair is not a matched-node speedup. There were no memory waits, forced grants
or run-alone grants. Together the three chromosome gates cover 2,696 blocks.

With fractional fixed-K reductions and discovery's CPU handoff added, matched same-node discovery
pairs with identical kernel manifests measure:

- std_n80 chr8: 38.6 to 33.3 s (-13.7%);
- N320 chr8: 72.9 to 53.5 s (-26.6%);
- TroMau chr7: 45.0 to 32.7 s (-27.3%).

All 2,696 blocks are unchanged. In the full N-series pipelines, discovery on the five chromosomes takes
235 s at N=160 and 369 s at N=320, against 7,973 s and 10,266 s in 2.4. The whole pipeline takes 10.6,
14.6, 22.6 and 45.6 minutes at N = 80, 160, 320 and 640, a scaling exponent of about 0.8 (the N=80 run
includes first-use kernel compilation).

### Component measurements before the small-work refinement

The same-host 24-core warm adverse N320 chr8 block22 falls from **1,625.9 to 9.8 s**
and **54.4 to 2.1 GiB peak RSS**. Process CPU falls from 1,623.3 to 11.7 s. The early
K=1 control43 falls from 1.133 to 0.728 s; its CPU rises from 0.765 to 0.844 s and peak
RSS from 1.549 to 1.591 GiB. Both preserve every compared scientific field, with zero
observed continuous drift. Each process was warmed on its own block; RSS includes
warmup/evidence. Baseline per-lift timing wrappers add overhead, so ratios are indicative.

The exact remedy completes all **580 N320 chr8 blocks in 76.6 s on 76 Ice Lake cores**,
with 4,766.4 task CPU-s and 30.32 GiB sampled summed process RSS (shared pages may be
counted repeatedly). All saved scientific decisions, families, calls and omissions agree;
maximum observed continuous drift is zero. Only diagnostic context suffixes differ.
The saved **1,564.7 s** used **112 AMD EPYC cores** and different timer boundaries;
this is a qualified historical comparison, not a matched-hardware speedup.

The last 68 results finish in 6.38 s after the 512th, versus the historical 1,501 s tail.
All 76 workers become active, and dynamic grants rise to 16 threads as workers finish.
Measured whole-run task CPU/wall is 62.24 occupied CPU equivalents; no memory waits,
forced grants or run-alone grants occurred. This does not claim perfect saturation.
Parallel geometry timing is a mixed parent envelope: unused serial phase buckets are
not zero worker costs, and overlapping task-wall sums are not exclusive elapsed times.

These component measurements precede the population-half proposal-family update.
Integrated canaries check their composition; final scaling and downstream validation
remain separate gates.

## Default-on A256 measurements and scaling limits


Let N be samples, M retained markers, L markers per block, K founders, R <= NK
residual records (default responsibility threshold further bounds R by a constant
times N), A=256, P<=K^2 partner routes, U searched panels, and I fit iterations.
Assume the genome/block width, cross design/K and computational budgets are fixed
when describing scaling with N. Let F(A) denote hierarchy/clique work on at most A
vertices; it can be exponential in A and is a potentially large fixed constant.
Exact packed enumeration/counting and size-class screening apply to ordinary and lifted
components in 2.5. Actual anchor-clique counts admit whole components; packed unions and
collision-checked deduplication count unique original-record size classes before Python
groups are materialised. This removes base24's A256 lifted-path limitation while preserving
the 64/128 complete-class policy, nearest-anchor ties and multiplicities. With C admitted
cliques, packed storage scales with C ceil(R/64), plus hash storage; graph-dependent
traversal/lifting remains and no new clique cap or polynomial worst-case claim is introduced.
Dense residual pools, a single pool or pools with one live budget thread run inline
with shared caches. Component jobs at most 4,096 estimated traversal steps and 1 MiB
scratch use the block reservation and run inline; these thresholds select execution only. Substantial independent work uses live dynamic CPU sharing;
large counted storage, scratch and decoding retain explicit memory reservations. Parallel
cache snapshots are confined to that concurrent path. Available cores pass to remaining
work without a fixed split or named-contig schedule; execution thresholds do not alter
component admission, class budgets or scientific outputs.

| Phase | Dense reference | Enabled A256 |
|---|---|---|
| Input/evidence tables | O(NM) | unchanged |
| K1 descriptors and geometry per block | O(N^2 L) distances and quadratic arrays | O(N L log N + A N log N + A N L + F(A)); O(N(L+A)+A^2) arrays |
| Fixed-K A/H fitting | leading O(I N L K^2), plus K-dependent solves | unchanged |
| Cavity score | O(N L f(K,I)) with held-out patches | unchanged |
| Residual records | O(N K^2 L), O(RL+NK^2) storage | unchanged |
| Each residual pool | O(R^2 ceil(L/64)) distances, quadratic hierarchy storage, potentially cubic diameter work and exponential cliques | O(R L log R + A R log R + R A ceil(L/64) + F(A)), plus graph-dependent packed lifting/deduplication, O(C ceil(R/64)) packed storage and hash storage; Python groups/consensus for retained classes |
| Partner routes | corresponding dense terms summed over P<=K^2 routes, each at most N records | corresponding bounded terms summed over the same routes; small exact routes also have at most A records |
| Search/release | preceding costs multiplied by U and M/L blocks; release O(NKL) per block | same search policy, altered proposals; O(N log N) dependence at fixed K,L,A,U,I |

The prototype and duplicate sorting produce N log N terms: this is not literally
an O(NA)-only algorithm. For fixed K, L, A and search/iteration budgets, the formal sample dependence
is below N^1.2, with graph-dependent combinatorial constants. This does not establish
practical scaling: historical N320 timed out, and the 2.5 chromosome remedy does not
establish the measured whole-pipeline target. No claim is made if K, genome size, solver iterations or admitted
search count grow with N, or that F(256) is harmless on all possible graphs.
The memory planner switches from dense squares to conservative linear record-by-A
scratch reservations above the activation boundary. The same exact1 digest keys
avoid retained quadratic key bytes. Caches remain bounded or depend on fixed
search budgets; reported process RSS includes Python/JIT allocator high-water.


The historical complete-block pilot exponent is 0.330 for N>=320 (wall), not a whole-genome guarantee. Residual activation depends on record count R>256; small sample count alone does not establish inactivity. Integrated N-series results are required.

Pedigree acceptance now requires linear expensive exact M1/M2/decision/resampling work with bounded plausible candidates per child. Any all-pairs screen must remain cheap: project its measured per-pair cost to N=10,000 on matching hardware and report its share of the total-runtime fit, with a target at most10%. No dense [N,N] or [C,N,N] arrays may remain outside compact screen storage. The current integrated dense screen is not claimed to meet this requirement. The N-series reports screen wall/CPU per directed dyad and its observed/projection share separately; projections from four points are descriptive and must not be treated as asymptotic proof.

## Certified pedigree screen

The default pedigree stage uses a certified selective M1 screen. Unproved
candidates receive exact HMM scores; the genome-wide floor and complete retained
parent-pair panel are unchanged. Raw component scores are computed where a missing
parent requires the existing M2 fallback. Compact screen and fixed-score checkpoints
are distinct from raw reference scores, avoiding a second genome-floor transform.

On the matched 112-core EPYC 7763 simulation screen, exact computation took
322.982 seconds and 29,833.15 CPU seconds; selective computation took 121.286
seconds and 3,480.03 CPU seconds (2.663 times wall and 8.573 times CPU speedup).
The screen avoided 96.802% of exact chromosome-dyad queries. This is screen timing,
not whole-pipeline scaling. Dense downstream decision, exposure, cavity and
resampling arrays remain; the full memory and scaling requirements are not met.
The observed bound cost of 18.3669 ms CPU per full-genome pair projects to about
4.56 hours at N=10,000 under ideal 112-core utilization. Its fraction of a matched
whole-pipeline runtime awaits the N-series; do not compare different genome spans.

## Follow-up performance policy

Sparse downstream pedigree decisions, exposure, cavity/path and resampling remain
necessary to eliminate dense quadratic arrays. Full N80/160/320/640 validation and
omitted release controls remain follow-up. After2.3, the owner permits a few ULP of
floating-point difference for SIMD/FMA or reassociated reductions only when released
decisions remain unchanged. That policy is not used for2.3 identity swaps. The C8
orientation estimator is deferred to2.4; honest phase breaks remain the2.3 output.
