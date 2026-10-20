# Performance and resource use

## 2.3 integration measurements

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
its ideal 112-core floor from contig CPU is 987.59 seconds. Stage6ab whole-stage timing
is pending at freeze unless reported below. Eight passing std/stress identity controls
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
C1 X). N80 completed in 327.576 seconds; N320 and two-point exponents are pending
at freeze unless reported below. N160/N640 remain unmeasured unless separately
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

Evidence snapshot: **2026-10-20T08:22:56.652274+01:00**. At the evidence freeze (2026-10-20T08:22:56.652274+01:00), the exact terminal NOT_RUN list is empty: no deadline guard has yet skipped a task. Pending or running cells are uncompleted, not passes. The final report must add any later NOT_RUN records without changing this frozen transfer.

Std reference carry-over is established for discovery by all 11,822 saved c3 blocks on eight contigs: K1 has at most 80 samples and the largest residual pool is 184, below 256. Partner pools are subsets of the global pool. The inactive branches retain the dense code; exact clique changes and C1/G7/G8 deltas use their accepted identity evidence. This is compositional biological carry-over of the c3 reference, not a fresh final-tree run or proof for perturbed inputs.

Actm3's saved c3 residual pools reach 292, 310 and 270 on the three checked contigs,
so A256 can activate. Its c3 result does **not** carry to the final tree. No c4 actm3
reference has completed; final actm3 evidence is component-only. TroMau completed
on c4 and carries biologically through the accepted exact swaps, with G8's explicit
interval-report contract. Completion is not a truth or reversal/noise pass.

The uncompleted truth references are stress_n80, lad_K20_n80, lad_K30_n80, f3_fam,
backcross_s6202, reallike6_nf_s3, actm3 and sim_s10000. The queued c4 std reference
has not started. Both std and stress still lack their final iid12_s1, reversal and
rename_s1 controls. N320 is running, N160/N640 are unmeasured and AB whole-stage
timing is pending. TroMau reversal, the broader zero loop and final Gate W remain
uncompleted. Later results belong in the campaign report; this document does not
anticipate their outcomes.

## Where the time goes

| stage | parallel unit | main cost |
| --- | --- | --- |
| evidence | contig (read in parallel windows); prepared one contig ahead of discovery | VCF decoding, read-model fits |
| discovery | block, in one persistent pool for all pending contigs | fixed-K fits and the reversible K search per block |
| local selection | block (path fits), component (context refits), inside a contig lane | path-model fits and the candidate bank; two context assemblies |
| assembly | joined batch (hierarchy), component (founder refinement), inside a contig lane | linker EM and certified panel searches; converged refinement fits and count decisions |
| painting | sample block, inside a contig lane | binned emissions and the diploid HMM |
| pedigree | chromosome (preparation, scoring, exclusion); genome-wide decisions are a barrier | pair scoring on the fixed candidate panel; bootstrap and leave-one-chromosome-out resampling |
| founder refit, family phase, recombination | chromosome | coordinate sweeps; family message passing and polishing; per-meiosis decoding |

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
  per pending contig, limited by memory. Each lane takes its contigs longest first. Leases start in proportion to
  each lane's work and grow as finished lanes release theirs: at contig starts, at the phase starts the stages mark,
  and inside pool phases.
- **Pools.** A pool task claims `floor(total / active)` threads plus a share of the remainder at its numerical
  boundaries; during a batch's start-up every worker has one thread until the batch's workers have started.
  Freed threads move to surviving tasks at their next boundary; a running Numba kernel never changes its team.
- **Shared arrays** are copied into shared memory once per pool, with at most
  `execution.shared_arrays.COPY_THREADS = 16` threads (the copy saturates memory bandwidth at about 16 threads).
- **Serial mode** runs one contig at a time with every core: less throughput on a large node, smaller units of
  durable progress ([usage.md](usage.md#short-allocations)).

`[execution] profile = true` prints per-phase timings and the thread-allocation trace of the leases. Every run
record holds exclusive stage timings (stages timed inside another stage on the same thread are subtracted, helper
threads are recorded under their own name) and per-contig summaries.

## Memory

One memory governor per run (`execution/memory.py`) keeps every process of the run within a memory limit by
admission, never by failing:

- **Limit.** The memory available when the run starts (the node's available memory, every enclosing cgroup's limit
  minus its working set, and the Slurm job's memory) minus a headroom of `execution.memory.HEADROOM_FRACTION = 0.1`
  (at least 4 GiB, at most half), and at most `[execution] memory_limit` when set.
- **Reservations.** Pools, chromosome executors, contig lanes and concurrent fits reserve each task's estimated
  peak bytes before starting it and release them when it ends; a pool's shared arrays are reserved for the pool's
  lifetime. A task starts when its bytes fit next to the reservations already held and into the memory the node has
  at that moment; otherwise it waits. Waiting requests nested in running work are served first (finishing started
  work before starting new work), then in arrival order; a smaller request may pass a waiting larger one only when
  both fit, so large tasks are not starved. Work nested d levels deep may fill the ledger only up to the limit minus
  `execution.memory.NESTED_HEADROOM_FRACTION = 0.2` / 2^(d - 1) of it (80% at the first level, 90% at the second,
  95% at the third, ...), so outer work that waits for its sub-tasks always leaves room for them.
- **Progress.** A task larger than the limit runs alone when nothing else holds memory; if every holder is itself
  waiting, the first waiting request is admitted over the limit so that work continues. Both are logged; the
  governor never deadlocks and never signals a process.
- **Shared memory of killed runs.** Every shared-memory segment the run creates is named after the run, and a new
  run removes this user's ledgers and segments of runs that have died (no live process holds them) before it reads
  the available memory, so a run killed with its allocation leaves no memory behind in `/dev/shm`.
- **Lanes.** While a lane works on a contig it holds about the contig's planning size (its discovery records, two
  copies of its genotype likelihoods and its observation mask); the lanes running together may hold at most
  `execution.contig_lanes.LANE_RESIDENT_SHARE = 0.5` of the limit, and everything inside a lane reserves its own
  bytes.
- **Stage estimates.** Each stage supplies per-task byte estimates from its inputs, conservative where unsure;
  kernels with large temporaries tile them under fixed byte budgets, so their results do not depend on free
  memory. The memory summary lines at the end of the run log and the run record's `memory` entry report the limit, reservations, waits and wait time,
  the peak reserved and peak used memory, and each task kind's estimate against its measured peak.
- **Phases.** Within a contig, local selection and assembly release their pools and arrays before painting starts,
  so their memory and painting's never overlap; executors return freed heap to the system (`malloc_trim`) after
  each task and at phase boundaries.

Pacing changes when work runs, never what it computes.

## Compiled kernels

Every `@njit(cache=True)` kernel compiles into a cache directory per code version: the key covers the whole
package source and its location, the Python, NumPy, Numba and llvmlite versions, the code-generation settings and
the CPU model, because Numba's own cache checks only each kernel's file timestamp. Workers of a run share the
directory. Between runs, compiled kernels persist in `~/.cache/haplotype-reconstruction/numba` (under
`$XDG_CACHE_HOME` when set; the `execution.kernel_cache.KEEP_KEYS = 6` most recently used code versions are kept):
a run starts from the stored snapshot of its code version, and for a new version it compiles every listed kernel
in a pool of the run's cores before its first scientific pool, longest first. A run publishes its cache when it
ends if it compiled anything. These are disposable compiled kernels, not scientific checkpoints; a store failure
costs only compilation time and is logged. `NUMBA_CACHE_DIR` sets the base of the per-version directories
([configuration.md](configuration.md#environment)).

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

## Default-on A256 measurements and scaling limits


Let N be samples, M retained markers, L markers per block, K founders, R <= NK
residual records (default responsibility threshold further bounds R by a constant
times N), A=256, P<=K^2 partner routes, U searched panels, and I fit iterations.
Assume the genome/block width, cross design/K and computational budgets are fixed
when describing scaling with N. Let F(A) denote hierarchy/clique work on at most A
vertices; it can be exponential in A and is a potentially large fixed constant.
Hotspot3 changes this constant by exact packed enumeration/counting and size-class
screening; it does not impose a new clique cap.

| Phase | Dense reference | Enabled A256 |
|---|---|---|
| Input/evidence tables | O(NM) | unchanged |
| K1 descriptors and geometry per block | O(N^2 L) distances and quadratic arrays | O(N L log N + A N log N + A N L + F(A)); O(N(L+A)+A^2) arrays |
| Fixed-K A/H fitting | leading O(I N L K^2), plus K-dependent solves | unchanged |
| Cavity score | O(N L f(K,I)) with held-out patches | unchanged |
| Residual records | O(N K^2 L), O(RL+NK^2) storage | unchanged |
| Each residual pool | O(R^2 ceil(L/64)) distances, quadratic hierarchy storage, potentially cubic diameter work and exponential cliques | O(R L log R + A R log R + R A ceil(L/64) + F(A)), plus group lifting/consensus linear in R at fixed admitted group budget/A |
| Partner routes | corresponding dense terms summed over P<=K^2 routes, each at most N records | corresponding bounded terms summed over the same routes; small exact routes also have at most A records |
| Search/release | preceding costs multiplied by U and M/L blocks; release O(NKL) per block | same search policy, altered proposals; O(N log N) dependence at fixed K,L,A,U,I |

The prototype and duplicate sorting produce N log N terms: this is not literally
an O(NA)-only algorithm. For fixed K, L and budgets, the asymptotic sample dependence
is below N^1.2. No claim is made if K, genome size, solver iterations or admitted
search count grow with N, or that F(256) is harmless on all possible graphs.
The memory planner switches from dense squares to conservative linear record-by-A
scratch reservations above the activation boundary. The same exact1 digest keys
avoid retained quadratic key bytes. Caches remain bounded or depend on fixed
search budgets; reported process RSS includes Python/JIT allocator high-water.


The complete-block pilot exponent is 0.330 for N>=320 (wall), not a whole-genome guarantee. Residual activation depends on record count R>256; small sample count alone does not establish inactivity. Integrated N-series results are required.

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
