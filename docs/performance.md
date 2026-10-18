# Performance and resource use

How a run uses its CPUs and memory, what the compiled-kernel and checkpoint caches do, why results do not depend
on any of it, and what a whole-genome run costs. Runtime, peak memory and scaling are acceptance criteria of the
package, measured on the reference runs ([validation.md](validation.md)).

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
