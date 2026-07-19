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

## Before making changes

Before editing:

1. Read the relevant source files and their direct callers.
2. Run `git status --short`.
3. Inspect existing uncommitted changes in files that may be edited.
4. Do not overwrite, revert, or reformat unrelated user changes.
5. Identify the files expected to change.
6. Explain the intended behavioural effect.
7. For non-trivial work, provide a short implementation and validation plan before editing.

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
conda activate bio-env
```

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
conda env list
python -c "import PACKAGE; print(PACKAGE.__version__)"
```

Do not dump the complete environment or all environment variables unless specifically needed.

Known dependencies include scientific Python packages such as NumPy, Numba, SciPy, pandas, scikit-learn, matplotlib, tqdm, cyvcf2, blosc2, and multiprocess. Some workflows may also invoke tools such as `samtools` or `bcftools`.

Treat this list as descriptive rather than exhaustive. Verify imports and executable availability in the current environment.

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

When modifying parallel code, consider:

- process count;
- threads per process;
- Numba thread scope;
- memory multiplied across workers;
- serialisation and checkpoint loading;
- deterministic versus order-dependent behaviour;
- exceptions and failures inside workers;
- oversubscription;
- startup cost;
- behaviour under Slurm CPU affinity.

## HPC and Slurm rules

The project runs on Cambridge CSD3, commonly on the Sapphire Rapids partition.

The agent may be running inside an interactive `sintr` allocation. Determine the actual environment rather than assuming it from the hostname.

Useful checks include:

```bash
hostname
echo "${SLURM_JOB_ID:-unset}"
squeue -u "$USER"
```

The absence of `SLURM_JOB_ID` in a shell does not by itself prove that no allocation exists; a separately opened shell may be on the allocated node without inheriting the job environment.

### Resource rules

- Never run substantial computation on a login node.
- Lightweight file inspection, syntax checks, and small unit-like tests are acceptable where permitted.
- Begin with the smallest relevant resource request.
- Do not use the full 112-core node for a test unless the test specifically requires it.
- Estimate CPU count, memory, and likely runtime before expensive execution.
- Use explicit and reviewed `srun` commands for parallel work inside an allocation.
- Use a reviewed `sbatch` script for long-running or production-scale work.
- Do not submit, cancel, reprioritise, or modify a Slurm job without explicit user approval.
- Do not run repeated polling loops against Slurm.
- Do not launch the full pipeline unless explicitly requested.
- Do not assume `nproc` alone represents the complete Slurm allocation.

Before proposing an expensive command, explain:

1. what it will execute;
2. which inputs it will read;
3. which outputs it will create or overwrite;
4. requested CPUs;
5. expected memory;
6. expected runtime;
7. whether it can resume safely;
8. how it will be stopped or cleaned up if it fails.

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

Common generated artifacts include:

- `*.pkl.b2`
- `.pipeline_checkpoints*`
- `results_*`
- `logs/`
- VCF, BCF, BAM, and CRAM files
- validation CSVs and run summaries

Check `.gitignore` and `git status` before staging changes.

Never delete checkpoint or result directories without explicit approval. Checkpoint compatibility must be assessed before recommending a fresh run.

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
9. any generated files or jobs created.

Do not claim a change is scientifically validated solely because the code runs or a test passes.
