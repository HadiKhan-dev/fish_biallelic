# AGENTS.md

Instructions for AI coding agents and orchestrators working in this repository.

Read this file before inspecting, modifying, or executing the project. This is the canonical shared repository guidance across coding tools; tool-specific adapters may import or supplement it but must not contradict it. Follow the current repository code, data documentation, and explicit user instructions when they conflict with assumptions here.

## Mission and priority order

This is biologically, mathematically, and computationally demanding scientific
research software. Use deep reasoning where it matters: biological assumptions,
statistical models, likelihoods, identifiability, numerical methods, algorithm
design, simulation design, scientific validation, computational scalability,
and interpretation of results.

Do not spend that reasoning budget on speculative hardening of trusted internal
Python objects, exhaustive adversarial edge cases, generalized integrity
frameworks, or production-platform abstractions that are not required by the
requested workflow.

Apply this priority order:

1. Biological, statistical, and mathematical correctness in supported
   workflows.
2. Scientific validity, interpretable assumptions, and faithful interpretation
   of results.
3. Reproducibility and faithful interpretation of inputs and outputs.
4. A working, testable end-to-end implementation of the user's requested
   outcome.
5. Numerical correctness, stability, and downstream behavioural validity.
6. Computational performance, scalability, memory efficiency, and I/O
   efficiency on the intended real workload.
7. Safe, correct, and efficient use of CSD3 and Slurm resources.
8. Code clarity and maintainability.
9. Minimal implementation complexity.
10. Defensive hardening only when a real trust boundary or demonstrated
    supported failure requires it.

Performance is a first-class project requirement when it affects whether the
intended scientific workflow can complete at useful scale. Do not sacrifice
scientific correctness, reproducibility, or numerical validity for speed, but
do not treat runtime, peak memory, shared-filesystem I/O, or parallel scaling
as secondary cosmetic concerns.

Prefer localized and documented implementation complexity when all of the
following hold:

- a simpler implementation is demonstrably too slow, memory-heavy, or
  I/O-heavy for the intended workload;
- profiling, benchmarking, or clear algorithmic analysis identifies the
  bottleneck;
- the optimized implementation preserves the scientific model and acceptably
  equivalent downstream results;
- the complexity remains bounded and understandable;
- a clear validation or reference path remains available where practical.

When priorities conflict, explain and measure the trade-off rather than
silently optimizing a lower-priority concern.

## Project overview

This repository implements a scientific bioinformatics pipeline for founder-haplotype reconstruction and local-ancestry inference from low-coverage sequencing of experimental crosses.

Given a multi-sample VCF or BCF, the pipeline discovers founder haplotypes within marker blocks, assembles them across chromosomes, paints offspring as diploid mosaics of founder haplotypes, infers pedigree relationships, corrects phase, and derives recombination maps.

The codebase includes the `bhd_*` block-haplotype-discovery modules and several pipeline entry points. Statistical and biological correctness remain paramount, while material runtime, memory, I/O, and scaling constraints are first-class requirements for the real workflow and may justify localized, validated complexity.

The current primary real-data workflow is associated with the cichlid pipeline in `pipeline_tropheops.py`. Verify this against the current repository before assuming it remains the primary entry point.

## Task contract and scope control

At the start of non-trivial work, identify:

- the requested outcome;
- the supported execution path involved;
- the files likely to change;
- the scientific quantities or behaviours that may change;
- acceptance criteria;
- explicit non-goals;
- the smallest meaningful validation;
- performance, memory, I/O, or scaling acceptance criteria when they
  materially affect the intended workflow.

If the request is materially ambiguous, ask a focused question before implementing. Do not turn an implementation task into a broad audit, redesign, security review, provenance overhaul, or repository-wide cleanup.

Do not broaden scope merely because nearby code could be improved. An unrelated concern may be reported briefly, but do not investigate or fix it unless it is severe enough to invalidate the requested work or the user authorizes the expansion.

Stop when the requested behaviour works, the agreed validation passes, and the completion report is ready. Do not add a final speculative-hardening pass, optional abstraction layer, or unrelated refactor.

## Pragmatic engineering and threat model

This repository is trusted scientific research code used by trusted project members with trusted in-process Python callers. It is not a hostile multi-tenant service and does not use internal dataclass objects as a security boundary.

Continue to protect against real operational hazards such as data loss, destructive shell commands, leaked credentials, unsafe handling of genuinely external text in shell commands, corrupted or partial scientific outputs, and incorrect scientific conclusions. However:

- Do not defend against deliberate misuse of `dataclasses.replace`, monkey-patching, manual mutation of frozen objects, forged internal hashes, hostile pickle construction, or callers intentionally violating internal APIs unless such a path is part of supported project behaviour.
- Do not introduce tamper-evident object graphs, parallel identity systems, all-field integrity digests, generalized canonical serialization, schema registries, immutable wrappers, or fail-closed validation frameworks without a concrete current requirement.
- Treat hashes and digests as cache keys, content identities, checkpoint compatibility markers, or provenance aids according to their documented use. Do not silently reinterpret them as cryptographic integrity boundaries.
- Do not add adversarial tests for unsupported internal states merely because Python can construct them.
- Do not use terms such as “tampering”, “attack”, “trust boundary”, “fail closed”, or “integrity violation” for ordinary trusted in-process misuse unless an actual security boundary exists.

Prefer the simplest implementation that remains scientifically correct, operationally reliable, and sufficiently performant for the intended workload. Do not choose simplicity over a demonstrated material runtime, memory, I/O, or scaling requirement; keep any necessary complexity localized, measured, documented, and validated.

## Defect threshold

Before calling something a defect and changing code for it, establish all of the following where practical:

1. A supported or realistically accidental execution path reaches the condition.
2. A concrete call site, input shape, checkpoint state, or user action can trigger it.
3. The consequence is observable: incorrect scientific output, incorrect branch or selection, crash, data loss, broken documented behaviour, reproducibility failure, or material resource waste.
4. A reproducer, failing test, trace, or direct code-path evidence supports the claim.

Classify findings as:

- **Scientific defect:** can alter biological/statistical conclusions or scientifically meaningful outputs in supported use. Investigate rigorously.
- **Operational defect:** causes a real crash, hang, corruption, lost work, or unusable supported workflow. Fix with the smallest reliable change.
- **Maintainability issue:** concrete complexity or duplication that is already impeding current work. Address only when in scope.
- **Speculative hardening:** requires deliberate unsupported misuse, a hostile in-process caller, hypothetical future requirements, or no concrete consequence. Do not implement; mention only if useful.

A violated theoretical invariant is not automatically a defect. A digest that can be retained while a trusted caller deliberately replaces unrelated dataclass fields is not a defect unless normal project code can do this accidentally and the stale digest produces a real consequence.

Do not create tests whose only purpose is to force unsupported states and then use those tests to justify new production machinery.

## Scientific reasoning policy

Concentrate deep reasoning on:

- biological plausibility and explicit biological assumptions;
- probabilistic and statistical formulation;
- likelihoods, priors, objectives, thresholds, and identifiability;
- genotype likelihoods, low-coverage uncertainty, haplotype ambiguity, pedigree ambiguity, phase, recombination, and missing data;
- mathematical equivalence versus behavioural equivalence;
- numerical stability and consequences of floating-point changes;
- simulation designs with known truth;
- negative controls, sensitivity analysis, calibration, and uncertainty;
- algorithmic complexity and scientifically valid parallel decomposition;
- interpretation of validation metrics and failure modes.

When proposing a scientific-model change, state:

1. the current mathematical or biological assumption;
2. the proposed assumption;
3. why the current behaviour is inadequate;
4. which outputs may change;
5. how the change will be validated independently;
6. what evidence would falsify the proposal.

Do not substitute more elaborate software structure for missing biological or mathematical justification.

## Orchestration and delegation

The scientific task may benefit from multiple agents, parallel investigation, or nested delegation. Use the active orchestrator's capabilities when they improve scientific reasoning, implementation quality, validation, or elapsed time.

- Decompose work around concrete biological, mathematical, implementation, code-path, or validation questions.
- Give delegated agents enough context to preserve the task contract, scientific assumptions, relevant paths, acceptance criteria, and resource constraints.
- Use specialist roles such as `scientific-modeler`, `code-path-explorer`, `scientific-validator`, and `pragmatic-implementer` when useful; equivalent built-in roles are acceptable.
- The coordinating agent must synthesize results, reconcile conflicting assumptions, distinguish evidence from inference, and remain accountable for the final implementation and report.
- Delegation does not change the task scope or the repository threat model. Work proposed by any agent must still satisfy the defect threshold and scientific-validation rules in this file.
- Return durable outputs: relevant edits, test artifacts, concise findings, assumptions, uncertainty, and the exact next action. Do not rely on unreported intermediate reasoning as the only record of work.

No repository-level concurrency cap or delegation-depth rule is imposed here. Use the orchestrator and available compute responsibly, respecting the HPC resource rules below.

## Core operating rules

1. Preserve the mathematical and statistical meaning of the implementation.
2. Do not silently change algorithms, objective functions, convergence criteria, filtering rules, thresholds, priors, likelihood calculations, or biological assumptions.
3. Distinguish numerical implementation changes from scientific-model changes.
4. Bit-for-bit floating-point identity is not required.
5. Differences of up to a few hundred ULP are acceptable when caused solely by mathematically equivalent evaluation order, vectorisation, parallel reduction order, or equivalent library implementations.
6. A numerically small difference is not automatically harmless. If it changes a threshold comparison, branch, convergence outcome, selected haplotype, inferred pedigree, phase assignment, validation metric, or other scientific result, report it as a behavioural change.
7. Do not assume an improved headline metric proves correctness.
8. When uncertain whether behaviour changed, report the uncertainty and validate it.
9. Prefer narrowly scoped changes over repository-wide refactoring.
10. Do not remove apparently unused code until callers, dynamic imports, checkpoint compatibility, and diagnostic uses have been checked.
11. Do not change code solely to make a test or metric pass.
12. State assumptions explicitly.
13. Do not present a hypothesis, naming convention, cohort label, or inferred relationship as biological ground truth.
14. Do not claim scientific validation solely because code executes or a test passes.
15. Do not implement optional hardening or abstractions after acceptance criteria are met.

## Pedigree-specific biological constraints

The real cichlid dataset does not currently have established individual-level trio ground truth unless explicit breeding records or metadata prove otherwise.

Whenever inspecting, designing, implementing, or validating pedigree inference:

- A generation or cohort label such as `G0`, `F1`, or `F2` does not identify an individual's parents.
- Do not treat the two sequenced G0 individuals as a known parental pair.
- One sequenced G0 individual may potentially be a parent of some pedigree members while the other is unrelated. Do not assume which relationships are valid without explicit evidence.
- Do not automatically include either G0 individual as a parent candidate. Include an individual only when eligibility rules, breeding records, metadata, or a user-approved design make that individual legitimate.
- Two sequenced individuals from the other species are outside the pedigree and must not be used as candidates, anchors, positive controls, or inferred relatives.
- Do not label candidate pairs or trios true, false, positive, negative, or decoy without independent ground truth.
- Do not calibrate thresholds, IBS0, Mendelian error, genotype-likelihood compatibility, founder-label consistency, or rankings against an assumed G0 pair.
- Distinguish species identity, cohort or generation, parent eligibility, known parentage, and inferred parentage.
- When ground truth is absent, validate with simulations, internal consistency, chromosome-level stability, resampling, legitimate negative controls, generation constraints, competing-candidate margins, and explicit ambiguous or unresolved outcomes.
- Preserve support for missing biological parents, single observed parents, ambiguous candidates, and unresolved individuals where scientifically appropriate.

If repository metadata or documentation appears to contradict these statements, stop and report the evidence before changing the interpretation.

## Before making changes

Before editing:

1. Read every applicable repository instruction file from the repository root to the target file, including nested `AGENTS.md`, `AGENTS.override.md`, `CLAUDE.md`, `CLAUDE.local.md`, and orchestrator-specific rule files that the active tool loads.
2. Read the relevant source files and direct callers.
3. Run `git status --short`.
4. Inspect existing uncommitted changes in files that may be edited.
5. Do not overwrite, revert, or reformat unrelated user changes.
6. Identify expected files and intended behavioural effect.
7. For non-trivial work, provide a short implementation and validation plan.
8. For a purported defect, satisfy the defect threshold above before implementing.

Do not broaden the task beyond the user's request without explaining why and receiving approval when the expansion is material.

## Repository structure

Important entry points currently include:

- `pipeline_tropheops.py` — primary real cichlid-cross workflow.
- `pipeline_real.py` — another real-cross workflow.
- `pipeline.py` — broader or simulation-oriented pipeline driver.
- `pedigree_sim_pipeline.py` — simulated end-to-end validation against known truth.
- `simulate_sequences.py` — sequence or read simulation.
- `recombination_map.py` — downstream recombination-map generation and CLI.

Important infrastructure currently includes:

- `bhd_config.py` — shared model and algorithm configuration.
- `thread_config.py` — numerical-library thread configuration.
- `dynamic_threads.py` — dynamic Numba thread allocation.
- `checkpoint_io.py` — compressed checkpoint I/O.
- `vcf_data_loader.py` — VCF/BCF loading and genotype-likelihood preparation.

Major stages include block-haplotype discovery, chimera resolution, refinement and residual discovery, hierarchical assembly, sample painting, pedigree inference, phase correction, and recombination-map generation.

This is a guide, not an authoritative inventory. Inspect the current repository before relying on filenames, stage numbers, or relationships.

## Configuration

Shared tunable model thresholds and feature flags generally belong in `bhd_config.py`.

Before adding a constant:

1. Search for an existing equivalent.
2. Check how related parameters are organised.
3. Confirm it is shared rather than entry-point-specific.
4. Preserve `bhd_config.py` as logic-free if that remains the convention.

Dataset paths, output paths, and experiment-specific selections should remain in the appropriate entry-point configuration unless the existing architecture indicates otherwise.

Do not scatter unexplained numerical constants through scientific modules. Low-level numerical sentinels and capability flags may remain in their owning modules where established.

## Environment

The working Conda environment is normally:

```bash
conda activate /rds/user/ahk39/hpc-work/conda_envs/bio-env
```

`conda activate bio-env` is acceptable only when it resolves to that environment. Verify rather than assuming.

Do not recreate, modify, upgrade, or install packages into this environment without explicit user approval. Never run `sudo`, system package installation, unreviewed `pip install` or `conda install`, shell-startup changes, or shared module changes.

Before diagnosing a dependency issue, use focused checks such as:

```bash
which python
python --version
python -c "import sys; print(sys.executable)"
conda env list
python -c "import PACKAGE; print(PACKAGE.__version__)"
```

Do not dump all environment variables or the complete environment unless specifically needed.

Known dependencies include NumPy, Numba, SciPy, pandas, scikit-learn, matplotlib, tqdm, cyvcf2, blosc2, and multiprocess. Workflows may also invoke `samtools` or `bcftools`. Verify current imports and executables.

## CSD3 coding-agent sandbox compatibility

A coding agent on CSD3 may already run inside a user namespace. A tool that adds another Bubblewrap layer can fail before execution with:

```text
bwrap: Creating new namespace failed: nesting depth or /proc/sys/user/max_*_namespaces exceeded (ENOSPC)
```

For this repository:

- Treat this exact `ENOSPC` as a namespace-creation failure, not disk exhaustion.
- Run ordinary commands through the active orchestrator's configured workspace sandbox using normal/default execution. In Codex this may be named `csd3-workspace`; other orchestrators may expose a different sandbox name or no additional sandbox.
- Do not request full access or escalation solely to bypass it.
- Unless a higher-priority instruction requires a dedicated patch helper, use a reviewed unified diff with system `patch`, inspect the result, and keep edits inside approved workspace roots.
- If a required patch helper fails once with this exact pre-execution error, use the normal permitted fallback rather than retrying namespaces.
- After editing, run normal validation plus `git diff --check`.
- If ordinary commands also fail with the same error, stop and report it. Do not bypass access controls.

`AGENTS.md` cannot disable an outer sandbox imposed by Codex, Claude Code, Kimi Code, a desktop app, a launcher, or CSD3.

## Concurrency and multiprocessing

The project uses process-level parallelism and Numba-accelerated numerical
code. Existing concurrency behaviour is intentional and must not be changed
casually.

- Preserve the established multiprocessing start method.
- Functions passed across worker boundaries must remain picklable.
- Worker callbacks should normally be defined at module scope. Do not place
  worker functions inside `if __name__ == "__main__":` or inside another
  function unless the current execution model explicitly supports it.
- Preserve safeguards against BLAS, OpenMP, MKL, and Numba oversubscription.
- In new entry points, inspect existing entry points to determine where
  `thread_config` must be imported relative to NumPy and Numba.
- Do not increase process counts or thread counts merely because more CPUs
  are allocated.
- Do not assume that a 112-core allocation means every operation should use
  112 cores.
- Explain the process/thread model before changing parallel execution.
- Avoid nested parallelism unless the total process-by-thread product is
  explicitly bounded.
- State the planned process count, threads per process, and total
  process-by-thread budget before combining multiprocessing, Numba, BLAS,
  OpenMP, MKL, or multithreaded command-line tools.
- Account for per-worker memory, duplicated arrays, deserialized checkpoints,
  temporary buffers, process startup, serialization, and shared-filesystem
  pressure before increasing concurrency.
- Choose process-pool chunk sizes and task granularity that avoid excessive
  scheduling, pickling, repeated checkpoint loading, and per-worker memory
  duplication.

When modifying parallel code, consider process count, threads per process,
Numba thread scope, numerical-library limits, memory multiplied across
workers, serialization and checkpoint loading, deterministic versus
order-dependent behaviour, exceptions inside workers, oversubscription,
startup cost, shared-filesystem pressure, and behaviour under Slurm CPU
affinity.

## HPC and Slurm rules

The project runs on Cambridge CSD3, commonly on Sapphire Rapids. The agent may be inside an interactive `sintr` allocation or may have reached an allocated node through a separate SSH connection. Determine the actual environment.

Useful checks include:

```bash
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-unset}"
printf 'SLURM_CPUS_ON_NODE=%s\n' "${SLURM_CPUS_ON_NODE:-unset}"
printf 'SLURM_CPUS_PER_TASK=%s\n' "${SLURM_CPUS_PER_TASK:-unset}"
printf 'SLURM_JOB_CPUS_PER_NODE=%s\n' "${SLURM_JOB_CPUS_PER_NODE:-unset}"
printf 'SLURM_MEM_PER_NODE=%s\n' "${SLURM_MEM_PER_NODE:-unset}"
printf 'SLURM_MEM_PER_CPU=%s\n' "${SLURM_MEM_PER_CPU:-unset}"
printf 'nproc=%s\n' "$(nproc)"
printf 'nproc_all=%s\n' "$(nproc --all)"
grep Cpus_allowed_list /proc/self/status
python -c 'import os; print("sched_affinity_cpus=", len(os.sched_getaffinity(0)))'
squeue -u "$USER"
```

Missing Slurm variables do not prove no allocation exists. If absent, inspect `squeue -u "$USER"`, match the hostname, ask when multiple jobs could match, and use `scontrol show job JOB_ID` for authoritative resources.

Do not infer the allocation from physical node size, `/proc/cpuinfo`, or `nproc --all`.

### Resource rules

- Never run substantial computation on a login node.
- Lightweight inspection, syntax checks, and small unit-like tests are acceptable where permitted.
- Begin with the smallest scientifically meaningful test.
- Estimate CPU, memory, I/O, and runtime before expensive execution.
- Use only allocated resources and reserve capacity for the agent, OS, filesystem, and coordination.
- Use reviewed `srun` commands where required and reviewed `sbatch` scripts for long or production work.
- Do not submit, cancel, reprioritize, or modify Slurm jobs without explicit approval.
- Do not launch the full pipeline unless explicitly requested.
- Do not run repeated polling loops against Slurm.

### Performance policy

Performance is a substantive acceptance criterion whenever runtime, peak
memory, I/O volume, or scaling determines whether the intended real-data
workflow is usable. Optimize measured or analytically established bottlenecks,
not hypothetical micro-costs, and validate that performance changes preserve
scientific behaviour.

Before non-trivial computation, classify the likely bottleneck as one or more
of:

- CPU-bound numerical work;
- decompression or compression;
- shared-filesystem I/O;
- memory bandwidth;
- memory capacity;
- process startup or serialization;
- an inherently serial algorithm.

Then apply these rules:

- Keep lightweight repository inspection, syntax checks, small metadata reads,
  and short tests single-threaded when parallel startup would cost more than
  it saves.
- For genuinely CPU-bound work, use bounded parallelism up to the verified
  allocation, not automatically the entire node.
- Start with a representative bounded benchmark and a conservative worker
  count, then scale only when timing demonstrates useful speedup.
- Prefer partitioning independent work by chromosome, contig, region, sample,
  block, replicate, or file when scientifically valid.
- Do not parallelize in a way that changes statistical dependence,
  random-number semantics, deterministic ordering, convergence, or output
  interpretation without reporting and validating the change.
- Avoid multiple workers repeatedly reading or decompressing the same large
  file from shared storage.
- Prefer indexed queries, bounded genomic regions, and streaming pipelines
  over full-file decompression and large intermediate files.
- Do not increase concurrency when the measured bottleneck is
  shared-filesystem throughput or memory bandwidth.
- Avoid nested multiprocessing plus threaded numerical libraries unless
  every thread pool is explicitly capped and the total process-by-thread
  product fits the verified allocation.
- Do not launch several expensive analyses concurrently unless their combined
  CPU, memory, and I/O requirements have been estimated and fit safely within
  the allocation.
- Reserve enough CPU and memory for the agent process, operating system,
  filesystem activity, and worker coordination rather than saturating the
  node merely because CPUs appear idle.

### Tool-specific performance guidance

- Check each tool's documentation and the repository's existing usage before
  assuming a `--threads`, `-@`, worker, or process option accelerates the
  expensive part.
- For `bcftools` and related HTS tools, thread options often accelerate
  compression or decompression more than filtering, parsing, or scientific
  computation. Verify the specific subcommand rather than assuming linear
  scaling.
- Prefer streaming compatible `bcftools` stages through pipes when this avoids
  unnecessary intermediate VCF files.
- Prefer BCF or uncompressed BCF streams where appropriate and already
  supported by the workflow.
- Use `bgzip -@ N` or equivalent parallel compression only when compression
  is the measured bottleneck and the command supports it.
- Do not use Numba for file I/O, decompression, subprocess orchestration, or
  small one-off loops.
- Use Numba for stable, repeatedly executed numerical kernels only after
  profiling identifies Python computation as a material bottleneck.
- When adding Numba parallel execution, verify that the target operation
  actually parallelizes, cap the Numba thread pool, preserve a clear
  reference implementation where practical, and compare numerical and
  downstream scientific results.
- For Python process pools, choose chunk sizes and worker counts that avoid
  excessive pickling, repeated checkpoint loading, and per-worker memory
  duplication.

### Benchmarking and observability

For performance-sensitive work:

- benchmark the smallest representative real-data subset that exercises the
  relevant code path;
- record the exact input subset, command, configuration, worker/thread count,
  elapsed time, and exit status;
- capture peak memory where practical, for example with `/usr/bin/time -v`;
- distinguish wall-clock speedup from increased total CPU time, memory
  consumption, or I/O load;
- compare at least two sensible worker counts before recommending a default;
- do not extrapolate full-pipeline runtime from an unrepresentative tiny case
  without stating the limitation;
- leave temporary benchmark outputs in an explicitly named safe location and
  report them;
- do not overwrite production results or checkpoints for benchmarking.

Before proposing or running an expensive command, explain:

1. what it will execute;
2. which inputs it will read;
3. which outputs it will create or overwrite;
4. whether it is expected to be CPU-bound, memory-bound, or I/O-bound;
5. the verified allocated CPUs and memory;
6. the proposed workers and threads per worker;
7. expected memory per worker and total memory;
8. expected runtime;
9. whether it can resume safely;
10. how it will be stopped or cleaned up if it fails;
11. the smallest representative benchmark to run first.

## Data and generated files

The data concerns cichlid fish rather than human participants, but datasets and outputs can be large.

- Inspect only the smallest amount needed.
- Prefer headers, indexes, sizes, counts, metadata, and bounded samples.
- Do not print thousands of variants or records.
- Do not recursively scan large storage trees without approval.
- Do not copy large datasets into the repository or invent paths.
- Do not commit datasets, checkpoints, logs, or generated results.
- Do not expose unnecessary genomic records, sample identifiers, logs, or large output to external model context.
- Summarize large results locally.

Common artifacts include `*.pkl.b2`, `.pipeline_checkpoints*`, `results_*`, `logs/`, VCF, BCF, BAM, CRAM, validation CSVs, and run summaries. Check `.gitignore` and `git status` before staging. Never delete checkpoint or result directories without explicit approval.

## Temporary files and interrupted allocations

Allocations can end abruptly. Treat unfinished writes and node-local temporary files as partial or lost.

- Store important work on shared project or home storage.
- Use node-local scratch only for reproducible disposable data.
- Write important outputs atomically where practical: temporary file, validate, then rename.
- After interruption, inspect `git status --short`, `git diff --stat`, `git diff --check`, changed and untracked files, and known temporary/output locations.
- Classify relevant outputs as complete, partial, corrupt, absent, or uncertain before overwriting.
- Do not rerun expensive work until the last confirmed completed step is identified.

## Running the pipeline

Verify the current entry point and configuration mechanism before changing or
running it. Do not edit a production configuration merely to test.

Prefer a small synthetic input, temporary configuration, dedicated test
script, copied entry-point configuration, limited contig or stage, or existing
self-test. Do not start a complete real-data run as exploration.

Checkpointed execution may resume from earlier stages. Before running against
existing checkpoints, confirm that the proposed code is compatible with their
schema, scientific semantics, configuration assumptions, and cached
identities. Before recommending a fresh run, assess checkpoint compatibility
and the cost of recomputation; do not recommend deleting or abandoning
checkpoints merely because compatibility analysis is inconvenient. Never
delete checkpoint or result directories without explicit approval.

## Code style

Match surrounding code.

- Keep scientific logic explicit and use descriptive names for model
  quantities.
- Preserve units and document them when unclear.
- Comment mathematical intent rather than syntax.
- Avoid broad formatting changes mixed with behavioural changes.
- Use small functions when they clarify the algorithm, but do not fragment
  hot numerical kernels.
- Preserve Numba-compatible types and control flow.
- Optimize established or analytically clear bottlenecks without obscuring
  scientific correctness.
- Avoid unmeasured micro-optimizations, but do not reject localized complexity
  when it is required to meet a demonstrated runtime, memory, I/O, or scaling
  requirement.
- Preserve a clear reference or validation path for complex optimized kernels
  where practical.
- Avoid new dependencies unless necessary and approved.
- Keep dataset paths out of deep scientific modules where possible.
- Duplication is acceptable when eliminating it would require a premature
  framework.
- Do not add wrappers, factories, registries, generalized validators, or
  parallel identity schemes for one speculative case.

## Validation

Search for existing tests before concluding none exist. Possible mechanisms include module self-tests, `recombination_map.py --selftest`, simulated crosses, pipeline validation, held-out founder comparisons, pair-reconstruction recall, and validation CSVs or summaries.

Choose validation according to the change.

### Non-numerical changes

Use focused execution paths, import or syntax checks, small fixtures, schema checks, and regression checks on affected behaviour.

### Numerical or scientific changes

Before running:

1. Identify which results may change and why.
2. Identify relevant metrics and establish a baseline where practical.
3. Hold data, configuration, seeds, and resources constant.
4. Compare primary and secondary metrics, failures, inferred haplotype counts, convergence, missingness, pedigree consistency, phase or switch behaviour, runtime, memory, warnings, and exceptions as relevant.

Equal or improved pair-reconstruction recall is supporting evidence, not sufficient proof. Validate floating-point differences and downstream consequences. Small differences are unacceptable when they cross thresholds, alter rankings, discrete choices, convergence, haplotypes, pedigrees, phase, or reported scientific outputs.

Report stochasticity, nondeterminism, and resource-dependent behaviour.

### Pedigree validation without real-data trio truth

- Do not report real-data accuracy, precision, recall, false positives, or false negatives against nonexistent truth.
- Use simulations for supervised validation and real data for consistency, stability, plausibility, and failure-mode analysis.
- Report candidate margins, informative-site counts, chromosome support, resampling stability, missing-parent states, ambiguity, and unresolved outcomes.
- Use legitimate negative controls and biologically impossible relationships where constraints are known.
- Avoid circular validation.
- Distinguish exploratory hypotheses from validated assignments.

Tests should cover supported behaviour and realistic regressions. Do not create adversarial internal-object mutation tests unless the project explicitly supports such construction or a real regression requires them.

## Git rules

The user controls history. Without explicit instruction, do not commit, push, pull, merge, rebase, reset, checkout/switch, amend, tag, stash, clean, or force any Git operation. Never discard local changes.

After editing:

```bash
git status --short
git diff --stat
git diff --check
```

Show or summarize the relevant diff. Do not stage generated data, checkpoints, logs, or results.

## Shell-command rules

Explain non-trivial commands before running them. Require explicit approval before installing or upgrading software, deleting or moving files, overwriting results, modifying the Conda environment, changing Git history, submitting or cancelling jobs, running a full pipeline, launching a long or expensive test, or recursively scanning large directories.

Avoid destructive commands such as `rm -rf`, `git clean`, `git reset --hard`, and broad wildcard deletion. Prefer bounded output such as:

```bash
tail -n 100 FILE
grep -n -C 5 PATTERN FILE
sed -n 'START,ENDp' FILE
```

Do not expose secrets, credentials, tokens, private keys, or unrelated environment variables.

## Completion report

At task end, report:

1. files changed;
2. behaviour changed;
3. scientific or mathematical assumptions affected;
4. commands and tests run with exact outcomes;
5. tests not run;
6. resource-intensive validation still recommended;
7. uncertainties and risks;
8. generated files or jobs;
9. workers, threads, CPUs, peak memory, and elapsed time for performance work;
10. whether outputs or conclusions depend on unverified biological assumptions;
11. any out-of-scope concern noted but deliberately not pursued.

Do not claim scientific validation solely because code runs or a test passes. Do not propose additional hardening after the requested task is complete unless the user asks for it.
