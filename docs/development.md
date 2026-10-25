# Development guide

How the package is organised, the contract every stage follows, the checkpoint store, the execution layer, and
the rules for changing the code. Paths are relative to the package `haplotype_reconstruction/`.

## Layout

| package | responsibility |
| --- | --- |
| `cli.py`, `__main__.py` | argument parsing only; builds the configuration and calls the commands |
| `config/` | the run configuration: `config/schema.py` (typed sections, defaults, descriptions), `config/load.py` (preset + TOML + options + `--set`, strict validation, resolved TOML), `config/presets/` (`tromau.toml`, `aulstu.toml`), `config/stages.py` (run configuration -> each stage's config, genetic maps) |
| `store/` | the checkpoint store: `store/codec.py` (payload encoding), `store/container.py` (append-only files), `store/store.py` (run file, contig files, work states, record scopes), `store/identity.py` (canonical digests, code and stage identities), `store/registry.py` (stage names in pipeline order) |
| `execution/` | process set-up and parallel work: `execution/bootstrap.py`, `execution/resources.py`, `execution/budget.py` (CPU leases), `execution/pools.py`, `execution/shared_arrays.py`, `execution/chromosome_workers.py`, `execution/contig_lanes.py`, `execution/memory.py` (memory governor, heap trimming), `execution/kernel_cache.py`, `execution/batch_queue.py`, `execution/run_record.py`, `execution/run_log.py` |
| `model/` | types shared by several stages: `model/evidence.py` (ContigReads, ContigEvidence), `model/local_panels.py` (LocalPanel), `model/components.py` (prepared panels, founder components, component panels, count calls), `model/genetic_map.py` |
| `maths/` | numerical kernels shared by stages: `maths/logspace.py` (log-space sums), `maths/chains.py` (binary-hypercube chain kernels: two-state and two-bit four-state forward-backward and Viterbi, the max-product butterfly step, run contraction; family phase and recombination), `maths/emissions.py` (founder-pair genotype classes and the robust log emission with its normalisation and skip rule; the linker, the joins and founder refinement) |
| `evidence/` ... `recombination/`, `export/` | the ten stages (below) |
| `simulation/` | generating code only: design, pedigree and meiosis, reads, the observation source, truth records |
| `pipeline/` | the driver (`pipeline/driver.py`), observation sources, the sample table and eligibility, one contig's work (`pipeline/contig_work.py`), status, published-table hook |
| `reports/`, `evaluation/`, `templates/` | published tables, evaluation against simulation truth, template building and the example |

Dependencies point one way. Infrastructure (`config/`, `store/`, `execution/`) is below the shared types
(`model/`, `maths/`), which are below the stages; the stages import in pipeline order (evidence < discovery <
assembly < local_selection < painting < pedigree < founder_refit < family_phase < recombination: local selection
calls assembly for its context assemblies, and the final assembly is wrapped by `pipeline/contig_work.py`, so
assembly never imports local selection). The export stage reads the other stages' records only. `pipeline/`,
`cli.py`, `reports/`, `export/`, `evaluation/` and `templates/` sit on top. A stage never imports a later stage, the driver, the CLI, `config/` or `os.environ`.
Importing the package configures the numerical libraries (one thread per BLAS/OpenMP pool, Numba's threading layer
and cache directory) and does nothing else; no module has other import-time effects.

## Stages

`store/registry.py` lists the stages in pipeline order with the packages whose code each executes:

| stage | scope | code identity covers | entry point |
| --- | --- | --- | --- |
| `evidence` | contig | `evidence` | `evidence.run_contig(store, source, contig, config, budget)` |
| `discovery` | contig | `discovery` | `discovery.run(store, contigs, config, budget, prepare=...)` |
| `local_selection` | contig | `local_selection`, `discovery`, `assembly` | `local_selection.run_contig(store, contig, config, assembly_config, genetic_map, budget, ...)` |
| `assembly` | contig | `assembly` | `assembly.assemble(inputs, config, role=..., records=..., budget=...)` |
| `painting` | contig | `painting` | `painting.run_contig(store, contig, config, genetic_map, budget)` |
| `pedigree` | genome | `pedigree`, `painting` | `pedigree.run(store, contigs, sample_ids, eligibility, config, genetic_maps, budget)` |
| `founder_refit` | contig | `founder_refit`, `painting` | `founder_refit.run(store, contigs, config, budget)` |
| `family_phase` | contig | `family_phase` | `family_phase.stage.run(store, contigs, edges, config, genetic_maps, budget)` |
| `recombination` | contig | `recombination`, `family_phase` | `recombination.stage.run(store, contigs, edges, config, genetic_maps, budget)` |
| `export` | contig | `export` | `export.stage.run(store, contigs, config, output, budget, synthetic=..., reference=...)` |

One name per stage is used everywhere: record prefix, `identity/<stage>`, `complete/<stage>`, output directory
`OUTPUT/<stage>/`, stop point and evaluation stage. Simulated inputs use the prefix `simulation/` (an input step,
not a stage).

Every stage package has the same parts:

- one public entry point taking typed inputs (frozen dataclasses of NumPy arrays whose docstrings give units,
  shapes, dtypes and meaning) and reading upstream products through the upstream stage's `records` API, never
  through published files;
- `<stage>/config.py`: a frozen dataclass whose defaults are the production values. These are model constants; only the
  fields the run configuration maps in `config/stages.py` are user settings
  ([configuration.md](configuration.md#internal-model-constants) lists both);
- `<stage>/records.py`: the stage's record keys, typed encode/decode (records are plain dicts of arrays and builtins, not
  pickled classes), its identity, and the readers other code uses: `complete(file)`, `progress(file)` (record keys
  and small plans only, for `haplotypes status`), and for founder stages `founder_panels(file)` (positions, calls
  per component or block, for evaluation and export).

The driver (`pipeline/driver.py`) runs: checks -> run start (cores, memory governor, kernel cache, log, run record,
`config.toml`) -> simulated inputs -> evidence and discovery (one block pool and one block queue for all pending contigs, the next
contig's evidence prepared on a helper thread) -> per contig in lanes: local selection -> final assembly ->
painting -> pedigree -> founder refit -> family phase -> recombination -> export (`[export] enabled`). The reports
layer writes the published tables at fixed points (`pipeline/outputs.py`); nothing reads them back. The export's
files are derived from the records: a changed export identity re-exports a contig (removing its earlier files)
instead of refusing the run, and no stage reads the export record.

## Checkpoint store

A checkpoint directory holds `run.ckpt` (genome-wide records), one `contigs/<contig>.ckpt` per contig, and
`work/` only while transient resume state is live.

**Container** (`store/container.py`). One file is a header (magic, format version, kind, name, file identity, CRC)
followed by records, each a header (magic, sequence, write time, stored and decoded sizes, payload layout, record
identity, key, CRC), a payload (`store/codec.py`: pickle protocol 5 with out-of-band buffers compressed by Blosc2
in chunks) and a trailer (magic, sequence, length, CRCs). Committed bytes are never modified; the latest complete
record of a key wins. A record counts only once its trailer is complete, so an interrupted writer leaves at most
an incomplete last record, which readers skip and the next writer truncates. Appends are serialised by a
per-process thread lock and `flock` on the file (coherent across nodes on the cluster file system) and fsynced;
readers take no lock and index record headers and trailers without decoding payloads. Each committed record has a
unique `identity` (32 hex digits in its `RecordInfo`): the value downstream stages bind as their upstream identity.

**Store** (`store/store.py`). `CheckpointStore(root)` gives `.run`, `.contig(name)`, `.contig_names()`,
`bind_identity(file, section, identity)` (put-once of `identity/<section>`; a different identity raises
`CheckpointMismatch`: "use a new output root"), `bind_manifest`, `mark_stage_complete`, `require(key, contigs)` and
`work(contig, name)` (atomic resume-only state under `work/`). `RecordScope(file, prefix)` is a picklable view of
one section that pool workers can write through. A directory holding other subdirectories than `contigs/` and
`work/` is refused.

**Identities** (`store/identity.py`). A stage's identity is `stage_identity(stage, schema=, config=, upstream=)`:
the stage name, its record schema version, its config (`asdict`), the record identities of the upstream records it
read, and the code identity (SHA-256 of the `.py` sources of the stage's packages plus `model/` and `maths/`,
computed once per process). Execution, store, pipeline and reporting code never enters a scientific identity, so
changing them never invalidates results; changing a stage's code does. The run record holds one digest of the
whole package source for provenance.

**Record keys.** Contig files:

| stage | keys |
| --- | --- |
| simulation | `simulation/reads` (simulated allele depths), `simulation/truth` (founders, alleles, painting, crossovers; read only by evaluation) |
| evidence | `evidence` (positions, counts, likelihoods, observed mask, discovery sites, founder exclusion, blocks, report) |
| discovery | `discovery/plan`, `discovery/blocks/<n:06d>` (chunks of 64 blocks in completion order) |
| local_selection | `local_selection/<round>/blocks/<n:06d>`, `.../memos/<n:06d>`, `.../proposals`, `.../selected`, `.../assembly/...` for rounds `initial`, `l1`, `l2` |
| assembly | `assembly/completion`, `assembly/l<k>/plan`, `assembly/l<k>/join`, `assembly/l<k>/refine/...`, `assembly/l<k>/refined`, `assembly/final`; `joins/<digest>` shared by every assembly of the contig |
| painting | `painting/summary`, `painting` |
| pedigree | `pedigree/prepared`, `pedigree/screen`, `pedigree/scores`, `pedigree/exclusion` |
| founder_refit | `founder_refit` |
| family_phase | `family_phase` (resume state `work/<contig>/family_phase.p5.b2` while solving) |
| recombination | `recombination` |
| export | `export` (the contig's exported files: identity and manifest entry) |

Each contig file also holds the `identity/<section>` records of its stages. `run.ckpt` holds `manifest/run`
(input kind, contigs, sample axis, source identity), `simulation/manifest` (simulations: seeds, contigs, the
observed and generated pedigrees), `complete/<stage>`, the pedigree product `pedigree` and the summaries
`family_phase` and `recombination`, with their identities.

## Execution

- **Process set-up** (`execution/bootstrap.py`). At import: one thread per numerical-library pool, Numba's
  threading layer `omp` with passive waiting, a kernel cache directory per code version. At run start
  (`start_run`): the Numba pool size (the run's cores), the forkserver preload list (every stage's modules), the
  profiling flag and the memory governor. Workers inherit these settings.
- **CPU leases** (`execution/budget.py`). One `CpuBudget` tree (run -> lane -> pool -> worker): a lane's lease grows
  toward its share of the remaining work at phase starts (`budget.phase(name)`); pool tasks claim
  `floor(total / active) + remainder` threads at numerical boundaries (`claim_threads()`); a start-up gate gives
  one thread per worker until the batch's workers have started. A running kernel never changes its team.
- **Pools and executors.** `BudgetedPool` is the one process pool (forkserver, non-daemonic so tasks may open
  pools; shared arrays copied once into shared memory and attached by every worker). `ChromosomeExecutor` runs one
  task per chromosome in persistent workers that keep per-chromosome state across phases. `execution/contig_lanes.py`
  runs concurrent contig lanes on top of it, in its dynamic mode: each worker starts with one of the longest tasks
  and takes the next from one queue, longest first.
- **Memory** (`execution/memory.py`). One governor per run keeps a ledger of the planned peak bytes of all running
  work in every process; constructs reserve their tasks' bytes and wait while they do not fit, so memory paces work
  instead of ending the run ([performance.md](performance.md#memory)). Stage code that allocates large memory
  outside the constructs wraps it in `reserve_memory(nbytes)`; kernels with large temporaries tile under fixed byte
  budgets, so results never depend on free memory. `release_memory()` (garbage collection and `malloc_trim`) runs
  at phase boundaries.
- **Kernel cache** (`execution/kernel_cache.py`): every `@njit(cache=True)` kernel is found automatically, kept
  between runs and compiled ahead of the first pool for a new code version.
- **Batch queue** (`execution/batch_queue.py`): stages describe independent batches as a `BatchKind`; helpers
  started with `haplotypes batch-worker` execute bundles; only the coordinator writes records.
- **Run records** (`execution/run_record.py`): per invocation, the resolved configuration, provenance, exclusive
  stage timings (helper threads included), per-contig product summaries and memory statistics.

## Rules for changes

- Scientific correctness first: models, thresholds, priors, release and tie rules change only to fix a
  demonstrated defect or to compute the same quantity correctly, with the effect measured and recorded in the
  changelog. A numerically small difference that changes a discrete outcome (a founder count, a pedigree call, a
  phase switch) is a behavioural change and must be explained.
- Results never depend on thread counts, lanes, memory pacing or completion order. Parallel kernels reduce
  per-sample (or per-site) rows in a fixed order; do not introduce team-size-dependent reductions or let results
  depend on which worker computed them. Byte budgets that bound a computation are fixed constants or come from the
  task's own reservation, never from the memory that happens to be free.
- Performance and memory are acceptance criteria: keep the algorithmic shortcuts of the hot kernels; supply
  byte estimates for new parallel work; tile large temporaries under fixed budgets. Nothing may scale exponentially
  in the founder count K (up to 30 per block); see the cost sections of the [methods](methods/README.md).
- Checks belong at real boundaries (inputs, decoded records); trusted internal objects are not re-validated.
  Internal invariants live in tests.
- Settings: a new user setting is a field of `config/schema.py` (kind, default, one-line description) mapped in
  `config/stages.py`; model constants are fields of the stage's `<stage>/config.py` with a description in its docstring or a
  comment above the field. Regenerate the reference with `python docs/generate_configuration.py`. Nothing is read
  from the environment.
- A changed record layout bumps the stage's schema version in its `<stage>/records.py`; identities then keep old and new
  results apart.

## Documentation

`docs/configuration.md` is generated from the code (`python docs/generate_configuration.py`). `python
docs/check_docs.py` checks every page against the code: links and anchors, configuration keys and values,
command lines and options, the defaults stated for internal constants (written as `RootConfig.path = value`, for
example `PaintingConfig.minimum_release_posterior = 0.9`), module paths, that the generated reference is current
and that every environment variable the code reads is documented. Run both after changing settings, commands,
constants or pages. Pages describe the code as it is; history belongs in the changelog and the migration guide.

## Stopping processes safely

On shared or allocated nodes, stop only processes you started. A user-wide kill command can end the allocation
itself, and with it every job and session running there.

- Never build a list of processes to signal from a user-wide selection: no `ps -u "$USER"`, `pgrep -u`,
  `pkill -u`, `killall` or `kill -1`. `procps` combines selection options as a union, so `ps -u "$USER" --ppid PID`
  lists every process of the user, not the children of `PID`. Use `kill 0` or `kill -- -PGID` only for a process
  group you created yourself (for example with `setsid`) and whose ID you have checked.
- Do not use `pkill`: `pkill -f PATTERN` also matches the shell running it and any unrelated process whose
  command line contains the pattern. List candidates with `pgrep -f PATTERN` or `ps -o pid,ppid,args -p PIDS`,
  check them, and signal only the verified PIDs.
- Record the PID (or a dedicated process group) of background work when you start it, and stop it through that
  identity. To stop a process tree, start from a verified PID and walk its descendants through parent links
  (`/proc/PID/stat` or `pgrep -P PID`, recursively), excluding your own shell and its ancestors.
- Never signal session infrastructure: `screen` or `tmux` sessions, `sshd`, `slurmstepd`, `systemd`, an
  allocation's login shell, or a runner hosting other work. Ending any of them ends the Slurm job.
- Print and inspect a computed kill list before sending the signal; if it holds anything but your own processes,
  do not send it.

Signalling a run's main process is enough to stop it (Ctrl-C, or SIGTERM to its PID): contig lanes stop with it,
pool workers end after their current task, and every record already written is kept for the resume.
