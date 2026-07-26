# AGENTS.md

Instructions for AI coding agents working in this repository.

Read this file before inspecting, modifying, or executing the project. Follow the current repository code and documentation when they conflict with assumptions in this file.

## Project overview

This repository implements a scientific bioinformatics pipeline for founder-haplotype reconstruction and local-ancestry inference from low-coverage sequencing of experimental crosses.

Given a multi-sample VCF or BCF, the pipeline discovers founder haplotypes within marker blocks, assembles them across chromosomes, paints offspring as diploid mosaics of founder haplotypes, infers pedigree relationships, corrects phase, and derives recombination maps.

The codebase includes the `bhd_*` block-haplotype-discovery modules and several pipeline entry points. Statistical and biological correctness take priority over stylistic simplification or marginal performance gains.

The current primary real-data workflow is associated with the cichlid pipeline in `pipeline_tropheops.py`. Verify this against the current repository before assuming it remains the primary entry point.

## Core operating rules

1. Preserve the mathematical and statistical meaning of the implementation.
2. Do not silently change algorithms, objective functions, convergence criteria, filtering rules, thresholds, priors, likelihood calculations, or biological assumptions.
3. Distinguish numerical implementation changes from scientific-model changes.
4. Bit-for-bit floating-point identity is not required.
5. Differences of up to a few hundred ULP are acceptable when they arise solely from numerically equivalent operations, such as changed evaluation order, vectorisation, parallel reduction order, or equivalent library implementations.
6. A numerically small difference is not automatically harmless. If it changes a threshold comparison, branch decision, convergence outcome, selected haplotype, inferred pedigree, phase assignment, validation metric, or other scientific result, treat it as a behavioural change and report it explicitly.
7. Do not assume that an improved headline metric proves that a change is correct.
8. When uncertain whether behaviour has changed, report the uncertainty and treat it as requiring validation.
9. Prefer narrowly scoped changes over repository-wide refactoring.
10. Do not remove apparently unused code until its callers, dynamic imports, checkpoint compatibility, and diagnostic uses have been checked.
11. Do not change code solely to make a test or validation metric pass.
12. State assumptions explicitly.
13. Do not present a hypothesis, naming convention, cohort label, or inferred relationship as biological ground truth.
14. Do not claim scientific validation solely because code executes or a test passes.

## Pedigree-specific biological constraints

The real cichlid dataset does not currently have established individual-level trio ground truth unless explicit breeding records or metadata prove otherwise.

Apply these rules whenever inspecting, designing, implementing, or validating pedigree inference:

- A generation or cohort label such as `G0`, `F1`, or `F2` does not identify an individual's parents.
- Do not treat the two sequenced G0 individuals as a known parental pair.
- One sequenced G0 individual may potentially be a parent of some pedigree members, while the other is unrelated. Do not assume which relationships are valid without explicit evidence.
- Do not automatically include either G0 individual as a parent candidate. Include an individual only when explicit eligibility rules, breeding records, metadata, or a user-approved design make that individual a legitimate candidate.
- Two sequenced individuals from the other species are outside the pedigree and must not be used as pedigree candidates, anchors, positive controls, or inferred relatives.
- Do not describe any candidate pair or trio as true, false, positive, negative, or decoy unless independent ground truth supports that label.
- Do not calibrate thresholds, IBS0, Mendelian error, genotype-likelihood compatibility, founder-label consistency, or ranking metrics against an assumed G0 pair.
- Distinguish clearly between species identity, cohort or generation, eligibility to be a parent, known parentage, and inferred parentage.
- When no ground truth exists, validate using simulations with known truth, internal consistency, chromosome-level stability, resampling, negative controls, generation constraints, competing-candidate margins, and explicit ambiguous or unresolved outcomes.
- Preserve support for missing biological parents, single observed parents, ambiguous candidates, and unresolved individuals where scientifically appropriate.

If repository metadata or documentation appears to contradict these statements, stop and report the evidence before changing the interpretation.

## Before making changes

Before editing:

1. Read every applicable `AGENTS.md` and `AGENTS.override.md` from the repository root down to the target file.
2. Read the relevant source files and their direct callers.
3. Run `git status --short`.
4. Inspect existing uncommitted changes in files that may be edited.
5. Do not overwrite, revert, or reformat unrelated user changes.
6. Identify the files expected to change.
7. Explain the intended behavioural effect.
8. For non-trivial work, provide a short implementation and validation plan before editing.

Do not broaden the task beyond the user's request without explaining why.

## Repository structure

Important entry points currently include:

- `pipeline_tropheops.py` — primary real cichlid-cross workflow.
- `pipeline_real.py` — another real-cross workflow.
- `pipeline.py` — broader or simulation-oriented pipeline driver.
- `pedigree_sim_pipeline.py` — simulated end-to-end validation against known truth.
- `simulate_sequences.py` — sequence or read simulation.
- `recombination_map.py` — downstream recombination-map generation and command-line interface.

Important infrastructure currently includes:

- `bhd_config.py` — shared model and algorithm configuration.
- `thread_config.py` — numerical-library thread configuration.
- `dynamic_threads.py` — dynamic Numba thread allocation.
- `checkpoint_io.py` — compressed checkpoint input/output.
- `vcf_data_loader.py` — VCF/BCF loading and genotype-likelihood preparation.

Major scientific stages include block-haplotype discovery, chimera resolution, refinement and residual discovery, hierarchical assembly, sample painting, pedigree inference, phase correction, and recombination-map generation.

This section is a guide, not an authoritative inventory. Inspect the current repository before relying on filenames, stage numbers, or relationships.

## Configuration

Shared tunable model thresholds and feature flags generally belong in `bhd_config.py`.

Before adding a new constant:

1. Search for an existing equivalent.
2. Check how related parameters are organised.
3. Confirm that the value is genuinely shared rather than entry-point-specific.
4. Preserve `bhd_config.py` as logic-free if that remains the repository convention.

Dataset paths, run-specific output paths, and experiment-specific selections should remain in the appropriate entry-point configuration unless the existing architecture indicates otherwise.

Do not scatter unexplained numerical constants through scientific modules.

Low-level numerical sentinels, implementation details, and capability flags may remain in their owning modules where that is the established design.

## Environment

The working Conda environment is normally:

```bash
conda activate /rds/user/ahk39/hpc-work/conda_envs/bio-env
```

`conda activate bio-env` is acceptable only when it resolves to that intended environment. Verify rather than assuming.

Do not recreate, modify, upgrade, or install packages into this environment without explicit user approval.

Never run:

- `sudo`
- system package installation
- unreviewed `pip install` or `conda install`
- changes to shell startup files
- changes to shared module configuration

Before diagnosing a dependency problem, inspect the active environment using focused commands such as:

```bash
which python
python --version
python -c "import sys; print(sys.executable)"
conda env list
python -c "import PACKAGE; print(PACKAGE.__version__)"
```

Do not dump the complete environment or all environment variables unless specifically needed.

Known dependencies include scientific Python packages such as NumPy, Numba, SciPy, pandas, scikit-learn, matplotlib, tqdm, cyvcf2, blosc2, and multiprocess. Some workflows may also invoke tools such as `samtools` or `bcftools`.

Treat this list as descriptive rather than exhaustive. Verify imports and executable availability in the current environment.

## CSD3 Codex sandbox compatibility

Codex on CSD3 may already be running inside a user namespace. A tool that adds another Bubblewrap (`bwrap`) layer can then fail before its command executes with:

```text
bwrap: Creating new namespace failed: nesting depth or /proc/sys/user/max_*_namespaces exceeded (ENOSPC)
```

For this repository on CSD3:

- Treat this exact `ENOSPC` message as a user-namespace creation failure, not as evidence that the project filesystem is out of disk space.
- Run ordinary commands through the configured `csd3-workspace` sandbox using normal/default execution. Do not request full access or sandbox escalation solely to work around this error.
- Unless a higher-priority instruction requires a particular editing tool, do not use a dedicated patch helper that adds another `bwrap` layer. Apply a reviewed unified diff with the system `patch` command through normal/default execution, keep the edit confined to approved workspace roots, and inspect the resulting diff.
- If a higher-priority instruction requires the dedicated patch helper, attempt it no more than once. After this exact pre-execution failure, use the normal `patch` fallback if permitted rather than repeatedly creating namespaces.
- After editing, run the repository's normal validation plus `git diff --check`. A successful fallback edit does not remove any scientific validation requirement.
- If ordinary commands also fail with the same namespace error, stop and report the failure. Do not bypass access controls; the Codex sandbox configuration or CSD3 namespace policy must be fixed outside the repository.

`AGENTS.md` controls agent behaviour only. It cannot disable an outer sandbox imposed by Codex, the desktop app, a launcher, or CSD3.

## Concurrency and multiprocessing

The project uses process-level parallelism and Numba-accelerated numerical code.

Existing concurrency behaviour is intentional and must not be changed casually.

Important rules:

- Preserve the established multiprocessing start method.
- Functions passed across a worker boundary must remain picklable.
- Worker callbacks should normally be defined at module scope.
- Do not place worker functions inside `if __name__ == "__main__":` or inside another function unless the current execution model explicitly supports it.
- Preserve safeguards against BLAS, OpenMP, MKL, and Numba oversubscription.
- In new entry points, inspect existing entry points to determine where `thread_config` must be imported relative to NumPy and Numba.
- Do not increase process counts or thread counts merely because more CPUs are allocated.
- Do not assume that a 112-core allocation means every operation should use 112 cores.
- Explain the process/thread model before changing parallel execution.
- Avoid nested parallelism unless the total process-by-thread product is explicitly bounded.
- Account for per-worker memory, duplicated arrays, deserialised checkpoints, and temporary buffers before increasing worker count.

When modifying parallel code, consider:

- process count;
- threads per process;
- Numba thread scope;
- BLAS, OpenMP, and MKL thread limits;
- memory multiplied across workers;
- serialisation and checkpoint loading;
- deterministic versus order-dependent behaviour;
- exceptions and failures inside workers;
- oversubscription;
- startup cost;
- shared-filesystem pressure;
- behaviour under Slurm CPU affinity.

## HPC and Slurm rules

The project runs on Cambridge CSD3, commonly on the Sapphire Rapids partition.

The agent may be running inside an interactive `sintr` allocation or may have reached an allocated node through a separate SSH connection. Determine the actual environment rather than assuming it from the hostname.

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

The absence of `SLURM_JOB_ID` in a shell does not by itself prove that no allocation exists; a separately opened shell may be on the allocated node without inheriting the job environment.

If Slurm variables are absent:

1. use `squeue -u "$USER"` to identify running allocations;
2. match the current hostname to the job's allocated node;
3. when more than one job could match, do not guess — ask the user which allocation owns the session;
4. use `scontrol show job JOB_ID` for authoritative CPU, node, and memory details where available.

Do not infer the usable allocation from `nproc --all`, `/proc/cpuinfo`, or physical node size. `nproc --all` can report all processors installed on the node rather than the CPUs available to the current process or allocated job.

### Resource rules

- Never run substantial computation on a login node.
- Lightweight file inspection, syntax checks, and small unit-like tests are acceptable where permitted.
- Begin with the smallest scientifically meaningful test.
- Do not use the full 112-core node for a test unless the test specifically requires it.
- Estimate CPU count, memory, I/O volume, and likely runtime before expensive execution.
- Use only CPUs and memory allocated to the current Slurm job.
- Use explicit and reviewed `srun` commands for parallel work inside an allocation when required by the site's execution model.
- Use a reviewed `sbatch` script for long-running or production-scale work.
- Do not submit, cancel, reprioritise, or modify a Slurm job without explicit user approval.
- Do not run repeated polling loops against Slurm.
- Do not launch the full pipeline unless explicitly requested.
- Do not assume `nproc` alone represents the complete Slurm allocation.
- Reserve enough CPU and memory capacity for the agent process, shell, operating system, filesystem activity, and worker coordination.
- Do not saturate the node merely because idle CPUs are visible.

### Performance and parallelism policy

Before any non-trivial computation, classify the likely bottleneck as one or more of:

- CPU-bound numerical work;
- decompression or compression;
- shared-filesystem I/O;
- memory bandwidth;
- memory capacity;
- process startup or serialisation;
- an inherently serial algorithm.

Then apply these rules:

- Keep lightweight repository inspection, syntax checks, small metadata reads, and short tests single-threaded when parallel startup would cost more than it saves.
- For genuinely CPU-bound work, use bounded parallelism up to the verified allocation, not automatically the entire node.
- Start with a representative bounded benchmark and a conservative worker count, then scale only when timing shows useful speedup.
- Prefer partitioning independent work by chromosome, contig, region, sample, block, replicate, or file when scientifically valid.
- Do not parallelise work in a way that changes statistical dependence, random-number semantics, deterministic ordering, convergence, or output interpretation without reporting and validating the change.
- Avoid multiple workers repeatedly reading or decompressing the same large file from shared storage.
- Prefer indexed queries, bounded genomic regions, and streaming pipelines over full-file decompression and large intermediate files.
- Do not increase concurrency when the measured bottleneck is shared-filesystem throughput or memory bandwidth.
- Avoid nested multiprocessing plus threaded numerical libraries unless all thread pools are explicitly capped.
- Before combining multiprocessing, Numba, BLAS, OpenMP, MKL, or multithreaded command-line tools, state the total planned process-by-thread count.
- Do not launch several expensive analyses concurrently unless their combined CPU, memory, and I/O requirements are estimated and fit safely within the allocation.

### Tool-specific performance guidance

- Check each tool's documentation and the repository's existing usage before assuming a `--threads`, `-@`, worker, or process option accelerates the expensive part.
- For `bcftools` and related HTS tools, thread options often accelerate compression or decompression more than filtering, parsing, or scientific computation. Verify the specific subcommand rather than assuming linear scaling.
- Prefer streaming compatible `bcftools` stages through pipes when this avoids unnecessary intermediate VCF files.
- Prefer BCF or uncompressed BCF streams where appropriate and already supported by the workflow.
- Use `bgzip -@ N` or equivalent parallel compression only when compression is the measured bottleneck and the command supports it.
- Do not use Numba for file I/O, decompression, subprocess orchestration, or small one-off loops.
- Use Numba only for stable, repeatedly executed numerical kernels after profiling identifies Python computation as a bottleneck.
- When adding Numba parallel execution, verify that the target operation actually parallelises, cap the Numba thread pool, preserve a clear reference implementation where practical, and compare numerical and downstream scientific results.
- For Python process pools, choose chunk sizes and worker counts that avoid excessive pickling, repeated checkpoint loading, and per-worker memory duplication.

### Benchmarking and observability

For performance-sensitive work:

- benchmark the smallest representative real-data subset that exercises the relevant code path;
- record the exact input subset, command, configuration, worker/thread count, elapsed time, and exit status;
- capture peak memory where practical, for example with `/usr/bin/time -v`;
- distinguish wall-clock speedup from increased total CPU time or I/O load;
- compare at least two sensible worker counts before recommending a default;
- do not extrapolate full-pipeline runtime from an unrepresentative tiny case without stating the limitation;
- leave temporary benchmark outputs in an explicitly named safe location and report them;
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

The biological data concerns cichlid fish rather than human participants. Human-subject restrictions are therefore not assumed.

Nevertheless, datasets and outputs can be very large. Avoid unnecessary scanning, copying, or display of:

- VCF or BCF files;
- CRAM or BAM files;
- large checkpoint files;
- large logs;
- complete result directories;
- reference assemblies.

Rules:

- Inspect only the smallest amount of data necessary for the task.
- Prefer metadata, headers, indexes, file sizes, record counts, or small bounded samples.
- Do not print thousands of variants or records into the conversation.
- Do not recursively scan large storage trees without approval.
- Do not copy large datasets into the repository.
- Do not invent dataset paths.
- Do not commit datasets, checkpoints, logs, or generated results.
- Do not send unnecessary raw genomic records, sample identifiers, logs, or large command output to external model context.
- Summarise large results locally and expose only the information needed for the task.

Common generated artifacts include:

- `*.pkl.b2`
- `.pipeline_checkpoints*`
- `results_*`
- `logs/`
- VCF, BCF, BAM, and CRAM files
- validation CSVs and run summaries

Check `.gitignore` and `git status` before staging changes.

Never delete checkpoint or result directories without explicit approval. Checkpoint compatibility must be assessed before recommending a fresh run.

## Temporary files and interrupted allocations

Compute allocations can end abruptly. Treat unfinished writes and node-local temporary files as potentially partial or lost.

- Prefer shared project or home storage for work that must survive allocation expiry.
- Use node-local temporary storage only for reproducible scratch data whose loss is acceptable.
- Write new important outputs atomically where practical: write to a temporary filename, validate, then rename into place.
- After an interrupted allocation, do not assume the final command or file write completed.
- Inspect `git status --short`, `git diff --stat`, `git diff --check`, changed files, untracked files, and explicitly used temporary/output locations before resuming.
- Classify relevant files as complete, partial, corrupt, absent, or uncertain before overwriting them.
- Do not rerun an expensive command until the last confirmed completed step and surviving outputs are identified.

## Running the pipeline

Most pipeline entry points appear to be configured using module-level configuration blocks and then executed directly. `recombination_map.py` has a command-line interface.

Verify the current entry point and configuration mechanism before changing or running it.

Do not edit a production configuration merely to conduct a test. Prefer:

- a small synthetic input;
- a temporary configuration;
- a dedicated test script;
- a copied entry-point configuration;
- an explicitly limited contig or stage;
- an existing self-test.

Do not start a complete real-data run as an exploratory test.

Checkpointed execution may resume from earlier stages. Before running against existing checkpoints, confirm that the proposed code remains compatible with their schema and semantics.

## Code style

Match the surrounding file.

General expectations:

- Keep scientific logic explicit.
- Use descriptive names for model quantities.
- Preserve units and document them when not obvious.
- Add comments explaining mathematical intent rather than restating syntax.
- Avoid broad formatting changes mixed with behavioural changes.
- Prefer small functions when they clarify the algorithm, but do not fragment hot numerical kernels unnecessarily.
- Preserve Numba-compatible types and control flow in compiled functions.
- Use vectorisation or Numba where it improves an established performance bottleneck without obscuring correctness.
- Avoid premature optimisation.
- Avoid introducing new dependencies unless necessary and approved.
- Keep paths out of deep scientific modules where possible.

## Validation

Do not assume a comprehensive `pytest` suite exists. Search before concluding that no formal tests are present.

Possible validation mechanisms include:

- module self-tests;
- `recombination_map.py --selftest`;
- simulated crosses with known truth;
- built-in pipeline validation;
- held-out founder comparisons;
- pair-reconstruction recall;
- other validation CSVs and summaries;
- focused import, compilation, or smoke tests.

Choose validation according to the change.

### For non-numerical changes

Examples include logging, argument handling, or file organisation.

Validate with:

- focused execution paths;
- import or syntax checks;
- small synthetic fixtures;
- output-schema checks;
- regression checks on affected behaviour.

### For numerical or scientific changes

Before running:

1. Identify which result may change.
2. State why it may change.
3. Identify relevant validation metrics.
4. Establish a baseline where practical.
5. Use the same data, configuration, seeds, and resource settings before and after.

Compare more than one signal where available:

- primary reconstruction metrics;
- secondary metrics;
- failure counts;
- number of inferred haplotypes;
- convergence behaviour;
- missingness;
- pedigree consistency;
- phase or switch behaviour;
- runtime and memory;
- warnings and exceptions.

Equal or improved pair-reconstruction recall is supporting evidence, but is not sufficient by itself. Also inspect relevant secondary metrics, inferred haplotype counts, convergence behaviour, pedigree consistency, phase corrections, warnings, failures, runtime, and memory.

Floating-point arrays may differ by up to a few hundred ULP where the implementation is mathematically equivalent. Validate both the numerical difference and its downstream consequences.

Small floating-point differences are not acceptable when they:

- cross thresholds;
- alter rankings;
- change discrete choices;
- change convergence;
- alter inferred haplotypes;
- alter pedigrees;
- alter phase assignments;
- affect reported scientific outputs.

Report stochasticity, nondeterminism, and resource-dependent behaviour.

### Pedigree validation without real-data trio truth

When individual-level parentage ground truth is absent:

- do not report accuracy, precision, recall, false positives, false negatives, or ranking against truth for the real dataset;
- use simulations with known truth for supervised validation;
- use the real dataset for internal consistency, stability, plausibility, and failure-mode analysis;
- report candidate margins, informative-site counts, chromosome-level support, resampling stability, missing-parent states, ambiguity, and unresolved outcomes;
- test negative controls and biologically impossible relationships where legitimate constraints are known;
- avoid circular validation in which inferred relationships are reused as truth to justify the same scoring model;
- distinguish exploratory biological hypotheses from validated pedigree assignments.

## Git rules

The user controls version history.

Do not perform any of the following without explicit instruction:

- commit;
- push;
- pull;
- merge;
- rebase;
- reset;
- checkout or switch branches;
- amend;
- tag;
- stash;
- clean;
- force any Git operation.

Never discard local changes.

After editing:

```bash
git status --short
git diff --stat
git diff --check
```

Show or summarise the relevant diff. Do not include unrelated changes in the task.

Do not stage generated data, checkpoints, logs, or results.

## Shell-command rules

Explain non-trivial commands before running them.

Require explicit user approval before:

- installing or upgrading software;
- deleting or moving files;
- overwriting existing results;
- modifying the Conda environment;
- changing Git history;
- submitting or cancelling jobs;
- running a full pipeline;
- launching a long or expensive test;
- recursively scanning large directories.

Avoid destructive commands such as `rm -rf`, `git clean`, `git reset --hard`, and broad wildcard deletion.

Prefer bounded commands and bounded output. For example:

```bash
tail -n 100 FILE
grep -n -C 5 PATTERN FILE
sed -n 'START,ENDp' FILE
```

Do not expose secrets, credentials, tokens, private keys, or unrelated environment variables.

## Completion report

At the end of a task, report:

1. files changed;
2. behaviour changed;
3. scientific or mathematical assumptions affected;
4. commands and tests run;
5. exact test outcomes;
6. tests not run;
7. resource-intensive validation still recommended;
8. known uncertainties or risks;
9. any generated files or jobs created;
10. CPUs, threads, workers, peak memory, and elapsed time for performance-sensitive work;
11. whether any outputs or conclusions depend on unverified biological assumptions.

Do not claim a change is scientifically validated solely because the code runs or a test passes.
