# AGENTS.md

Rules for every AI agent and orchestrator working in this repository: Codex, Claude Code (through `CLAUDE.md`, which
imports this file) and any other tool. This file is the single source of shared policy. Tool-specific files may add
to it but never contradict it.

Read it before inspecting, changing or running anything. "The owner" is the person who owns this repository and its
allocations. If an explicit instruction from the owner in the current conversation conflicts with this file, follow
the owner and point out the conflict. The file states rules; check facts about the code, data and cluster against
their current state.

## 1. Non-negotiables

These override cost, speed and convenience. Each is detailed in the section cited.

1. **Never end an allocation that hosts an agent.** Never cancel, shorten, resize or put a cleanup timer on the
   allocation running Codex, Claude Code or their runners. Never signal its `screen`, `tmux`, `sshd`, `slurmstepd`,
   `systemd`, login shell or runner (§6.1).
2. **Never stop processes through a user-wide selection.**
   - No `pkill`, `killall` or `kill -1`, and no target list built from `ps -u` or `pgrep -u`.
   - Signal only processes you started, or their verified descendants (§7.2).
3. **The owner controls Git history.** Without an explicit instruction:
   - never commit, push, pull, merge, rebase, reset, check out, switch, amend, tag, stash, clean or force;
   - never discard local changes (§4.8).
4. **Never delete checkpoints, results or other files without explicit approval**, except the provisional outputs
   of your own authorized attempt. Never overwrite accepted results (§5.1, §7.3).
5. **Never test on real AulStu data.** No acceptance runs, validation, benchmarks, scaling studies or projections on
   the real AulStu (astcal) cross. Use simulations, TroMau and bounded inputs (§3.1).
6. **Request SL4-CPU only** (§6.2-§6.4).
   - SL3-CPU, SL2-CPU and SL3-GPU allocations need the owner's explicit request at the time.
   - SL3 and SL2 CPUs together never exceed 448.
   - Running SL4 allocations are kept until they expire.
7. **Whole-genome runs use at least one full node.** This includes validation, resume checks and canaries (§6.5).
8. **Preserve scientific meaning.** Never silently change a model, objective, threshold, prior, likelihood, filter
   or biological assumption. "Same results" means scientifically identical results, not identical bits (§2.3).
9. **Never invent ground truth.** Cohort labels such as G0, F1 and F2 do not identify parents, and no sequenced G0
   individual is a known parent (§2.4).
10. **Confirm every run starts, and watch it until it ends.** Tasks fail fast and never hang (§5.3).
11. **Keep scheduler traffic low.** Poll `squeue` at most once every 120 seconds, and submit through at most one
    opportunity every five minutes (§6.6).
12. **Leave the environment alone and keep secrets private.** No installs or Conda changes without approval. Never
    print `.env` files, credentials or tokens (§3.4, §7.1).

## 2. Science

### 2.1 Mission and priorities

This is biologically, mathematically and computationally demanding research software. Spend deep reasoning on:
- biological assumptions;
- statistical models, likelihoods and identifiability;
- numerical methods and algorithm design;
- simulation design, validation, scalability and the interpretation of results.

Do not spend it on speculative hardening of trusted internal objects, exhaustive adversarial cases, integrity
frameworks, or platform abstractions that the workflow does not need.

Priority order:
1. Biological, statistical and mathematical correctness in supported workflows.
2. Scientific validity, interpretable assumptions and faithful interpretation of results.
3. Reproducibility and faithful interpretation of inputs and outputs.
4. A working, testable end-to-end implementation of the requested outcome.
5. Numerical correctness, stability and downstream behavioural validity.
6. Performance, scalability, memory and I/O efficiency on the real workload.
7. Safe, correct and efficient use of CSD3 and Slurm.
8. Clarity and maintainability.
9. Minimal implementation complexity.
10. Defensive hardening, only where a real trust boundary or a demonstrated supported failure requires it.

**Performance is a first-class requirement** whenever it decides whether the scientific workflow completes at
useful scale.
- Never trade correctness, reproducibility or numerical validity for speed.
- Equally, never treat runtime, memory, filesystem I/O or scaling as cosmetic.
- Localized, documented complexity is justified when all of these hold:
  - a simpler version is demonstrably too slow or too large;
  - profiling or analysis identifies the bottleneck;
  - the model and downstream results are preserved;
  - the complexity stays bounded and understandable;
  - a reference or validation path remains where practical.
- When priorities conflict, measure and explain the trade-off.

### 2.2 Scientific reasoning

Concentrate on:
- biological plausibility and probabilistic formulation;
- likelihoods, priors, objectives, thresholds and identifiability;
- genotype likelihoods at low coverage, haplotype and pedigree ambiguity, phase, recombination and missing data;
- mathematical versus behavioural equivalence, and numerical stability;
- simulations with known truth, negative controls, sensitivity, calibration and uncertainty;
- algorithmic complexity and valid parallel decomposition;
- the interpretation of metrics and failure modes.

A proposed scientific-model change states:
1. the current assumption;
2. the proposed assumption;
3. why the current behaviour is inadequate;
4. which outputs may change;
5. how it will be validated independently;
6. what evidence would falsify it.

More elaborate software never substitutes for missing biological or mathematical justification.

### 2.3 Preserving scientific meaning

1. Preserve the mathematical and statistical meaning of the implementation.
   - Never silently change algorithms, objectives, convergence criteria, filters, thresholds, priors, likelihoods
     or biological assumptions.
   - Keep numerical-implementation changes distinct from scientific-model changes.
2. **"Same results" means scientifically identical results.** The owner, 20 October 2026: "I want scientific
   result identicality, and if some bits shift around due to getting faster that's 100% perfectly fine".
   - These are the same released outputs and decisions:
     - called genotypes and missingness;
     - founder alleles and counts, and selected panels and haplotypes;
     - released ambiguity, and pedigree calls and support;
     - phase assignments and phase sets, and crossovers and intervals.
   - Continuous outputs agree to well below any scientifically meaningful scale.
   - Bit identity is required nowhere, including in refactors and speedups.
   - Floating-point differences from mathematically equivalent computation are acceptable. That covers another
     evaluation or reduction order, SIMD or FMA, parallel reductions, and equivalent libraries or kernels.
   - Never slow down or complicate code to preserve bits, and never gate performance work on byte equality.
     Compare the released outputs, and report the drift in key continuous quantities.
3. A small numerical difference is still a behavioural change, to be reported, if it changes any of: a threshold
   comparison, a branch, convergence, a selected haplotype, a pedigree, phase, a metric or another scientific result.
4. An improved headline metric does not prove correctness. When unsure whether behaviour changed, say so and
   validate it.
5. Never change code only to make a test or metric pass. Never claim scientific validation because code runs or a
   test passes.
6. State assumptions explicitly. Never present a hypothesis, naming convention, cohort label or inferred
   relationship as biological ground truth.
7. Prefer narrow changes to repository-wide refactoring. Before removing apparently unused code, check its callers,
   dynamic imports, checkpoint compatibility and diagnostic uses.

### 2.4 Pedigree constraints

The real cichlid data have no established individual-level trio truth unless breeding records or metadata prove
otherwise. In every pedigree task:
- A generation or cohort label (G0, F1, F2) does not identify anyone's parents.
- The TroMau cross has four sequenced G0 individuals, three plus one:
  - three AstCal females: fAstCal29, fAstCal30 and fAstCal31;
  - one TroMau male: fTroMau1.

  Every sequenced sample in the cross is labelled G0, F1 or F2.
- No sequenced G0 individual, or pairing of them, is a known parent. Any of them may be a parent of some members and
  unrelated to others.
- An individual is a parent candidate only when eligibility rules, breeding records, metadata or an owner-approved
  design make it legitimate.
- Individuals outside a cross's pedigree are never candidates, anchors, positive controls or inferred relatives.
- Never label pairs or trios true, false, positive, negative or decoy without independent truth.
- Never calibrate any of these against assumed G0 parentage: thresholds, IBS0, Mendelian error,
  genotype-likelihood compatibility, founder-label consistency or rankings.
- Keep species, cohort, parent eligibility, known parentage and inferred parentage distinct.
- Without truth, validate through:
  - simulations, internal consistency and chromosome-level stability;
  - resampling and legitimate negative controls;
  - generation constraints and competing-candidate margins;
  - explicit ambiguous or unresolved outcomes.
- Preserve support for missing, single-observed, ambiguous and unresolved parents.

If repository metadata or documentation seems to contradict this, stop and report the evidence before changing the
interpretation.

### 2.5 Design requirements

**One pipeline that works up to K = 30 founders** (owner, 5 October 2026).
- Today's data have about ten founders per block or fewer, but no stage may hide caps sized for small K that break or
  decide results at larger K.
- Nothing may be exponential in K, and no stage may cost more than O(K^3) in basic time complexity. Prefer better
  algorithms to budgets.
- Check K-dependence (caps, budgets, costs and integer widths) by code analysis and synthetic kernel benchmarks, not
  by full simulations unless asked.
- Keep one unified route. Remove superseded alternative modes and relict code, after the checks in §2.3.

### 2.6 Validation

Match validation to the change.
- **Non-numerical changes:** focused execution paths, import or syntax checks, small fixtures, schema checks and
  regression checks on the affected behaviour.
- **Numerical or scientific changes:**
  1. Identify which results may change, and why.
  2. Choose the metrics and a baseline. Use an existing one, or obtain it in parallel without delaying an isolated
     attempt.
  3. Hold data, configuration, seeds and resources constant.
  4. Compare:
     - primary and secondary metrics, and failures;
     - haplotype counts and convergence;
     - missingness, pedigree consistency, and phase and switches;
     - runtime, memory, warnings and exceptions.
- **Recall is not proof.** Equal or better pair-reconstruction recall supports a change but does not prove it.
- **Discrete changes are unacceptable unless intended and validated:** differences that cross thresholds or alter
  rankings, discrete choices, convergence, haplotypes, pedigrees, phase or reported outputs.
- **Report** stochasticity, nondeterminism and resource-dependent behaviour.
- **Known-truth validation:** `haplotypes run CONFIG` with `[input] kind = "simulation"`, then `haplotypes evaluate
  RUN`. Other mechanisms:
  - bounded synthetic inputs and held-out founder comparisons;
  - pair-reconstruction recall;
  - phase and genotype error counts, and validation summaries.

  Historical focused tests and fixtures are archived locally, outside the public package.
- **Pedigree without real-data truth:**
  - Never report real-data accuracy, precision, recall, false positives or false negatives against truth that
    does not exist.
  - Use simulations for supervised validation. Use real data for consistency, stability, plausibility and failure
    modes.
  - Report candidate margins, informative-site counts, chromosome support, resampling stability, missing-parent
    states, ambiguity and unresolved outcomes.
  - Use legitimate negative controls and biologically impossible relationships, and avoid circular validation.
  - Keep hypotheses separate from validated assignments.
- **Test scope:** tests cover supported behaviour and realistic regressions, not adversarial internal states
  (§4.2). How much to validate, and when, is set in §5.5.

## 3. The project

### 3.1 Overview

The package reconstructs founder haplotypes and local ancestry from low-coverage sequencing of experimental crosses.
From a multi-sample VCF or BCF it:
1. discovers founder haplotypes within marker blocks;
2. assembles them along chromosomes;
3. paints offspring as diploid mosaics of founders;
4. infers the pedigree and corrects phase;
5. estimates recombination maps.

It is one pipeline of stage packages run by the `haplotypes` command: evidence, discovery, local_selection, assembly,
painting, pedigree, founder_refit, family_phase, recombination and export.
- **The primary real workflow is the TroMau cichlid cross:** `haplotypes run CONFIG` with `preset = "tromau"`. The
  preset is `haplotype_reconstruction/config/presets/tromau.toml`, and the example is `configs/tropheops.toml`.
- **AulStu is never used for testing.** The owner, 17 October 2026: "real data aul stu is very very bloated, do not
  use it for any testing".
  - Its preset exists for production runs, which are the owner's decision.
  - Finished AulStu measurements may be cited as data.

### 3.2 Repository map

Entry points:
- `haplotypes` (`haplotype_reconstruction/cli.py`, also `python -m haplotype_reconstruction`), with the commands
  `run`, `status`, `recombination`, `evaluate`, `export`, `example`, `templates` and `batch-worker`.
- `haplotype_reconstruction/pipeline/driver.py`: fail-fast checks, stage order, resume and shards.
- Presets and example configurations:
  - `haplotype_reconstruction/config/presets/{tromau,aulstu}.toml`;
  - `configs/*.toml`, covering TroMau, AulStu, generic VCF and simulations.
- `haplotype_reconstruction/simulation/`: simulated end-to-end validation (§2.6).
- `haplotype_reconstruction/recombination/`: `haplotypes recombination RUN` reruns the stage.
- `haplotype_reconstruction/export/`: per-chromosome founder and phased-sample VCFs in `OUTPUT/export/`.
  `haplotypes export RUN DEST` re-exports.

Infrastructure, under `haplotype_reconstruction/`:
- `config/`:
  - the TOML schema (`schema.py`);
  - the loader and presets (`load.py`);
  - the mapping to each stage's frozen config (`stages.py`).
- `execution/`:
  - the CPU budget and process pools;
  - numerical-library and Numba set-up (`bootstrap.py`);
  - the memory governor (`memory.py`), run records and the kernel cache.
- `store/`: the checkpoint store, codecs and code identities. Each stage's `records.py` owns its checkpoint keys.
- `evidence/`: VCF/BCF loading, site and sample filters, the read model and genotype likelihoods.

**How the stages fit together.** `docs/methods/` documents each stage. In brief:
- discovery is missing-aware;
- local selection chooses paths with assembly feedback contexts;
- assembly is hierarchical (L1-L4) with joint founder completion;
- painting is component-local;
- pedigree inference uses the paintings and raw genotype likelihoods, with Tier B as the primary output;
- family phase refines phase conditioned on the pedigree;
- recombination estimates maps from the final phase.

Family phase and recombination never feed back upstream. The direction and family approximations depend on the
painting, so ambiguous, missing-parent and structured-missingness crosses need validation. Inspect the code before
relying on any filename or relationship given here.

### 3.3 Configuration

- **Where settings live:**
  - Tunable model thresholds live in the owning stage's frozen config dataclass, `<stage>/config.py`.
  - User-facing settings live in the TOML schema, mapped to the stage configs in `config/stages.py`.
  - Internal constants are listed in `docs/configuration.md`.
- **Before adding a constant:** search for an equivalent, follow how related parameters are organised, and confirm
  it is shared rather than specific to one entry point.
- **Keep configuration modules logic-free.** Regenerate `docs/configuration.md` with
  `docs/generate_configuration.py` whenever the schema, the CLI, a preset or a stage config changes.
- **Paths and selections:** dataset paths, output paths and experiment selections belong in entry-point
  configurations.
- **Numerical constants:** do not scatter unexplained ones through scientific modules. Established low-level
  sentinels and capability flags may stay in their modules.

### 3.4 Environment

- Use `conda activate /rds/user/ahk39/hpc-work/conda_envs/bio-env`. `conda activate bio-env` is acceptable only if
  it resolves there, so verify it.
- Never recreate, modify, upgrade or install into it without explicit approval. Never run `sudo` or system package
  installation, an unreviewed `pip install` or `conda install`, or changes to shell start-up files or shared modules.
- Diagnose dependency problems with focused checks, never by dumping the environment:
  - `which python` and `python --version`;
  - `python -c "import sys; print(sys.executable)"` and `conda env list`;
  - `python -c "import PACKAGE; print(PACKAGE.__version__)"`.
- Known dependencies: NumPy, Numba, SciPy, pandas, scikit-learn, matplotlib, tqdm, cyvcf2, blosc2 and multiprocess.
  Workflows may also call `samtools` or `bcftools`.

## 4. Engineering practice

### 4.1 Task contract and scope

At the start of non-trivial work, identify:
- the requested outcome and the supported execution path;
- the files likely to change;
- the scientific quantities or behaviours that may change;
- the acceptance criteria and the non-goals;
- the smallest meaningful validation;
- any performance, memory, I/O or scaling criteria that matter.

Then:
- **Ask** a focused question when a request is materially ambiguous.
- **Stay in scope.** Never widen a task into an audit, redesign, security review, provenance overhaul or cleanup.
  Report an unrelated concern briefly. Pursue it only if it invalidates the requested work or the owner agrees.
- **Match the process to the change** (owner, 5 October 2026).
  - Changes to the scientific model or decision rules (likelihoods, priors, counting or novelty criteria), and
    algorithm redesigns, get the full process: diagnosis, then a design with prototype runs, then one adversarial
    review.
  - Simple fixes get one focused implementation and its focused check.
  - Verify what changed rather than re-accepting everything from scratch.
- **Stop** when the requested behaviour works, the agreed validation passes and the report is ready. Add no final
  hardening pass, optional abstraction or unrelated refactor.

### 4.2 Threat model and defects

This is trusted research code, used by trusted project members with trusted in-process callers. Internal
dataclasses are not a security boundary.

Protect against real hazards:
- data loss, destructive commands and leaked credentials;
- unsafe handling of genuinely external text in shell commands;
- corrupted or partial outputs;
- wrong scientific conclusions.

But do not:
- defend against deliberate misuse unless that path is supported behaviour: `dataclasses.replace`,
  monkey-patching, mutating frozen objects, forged hashes, hostile pickles or deliberate API violations;
- add tamper-evident object graphs, parallel identity systems, all-field digests, generalized canonical
  serialization, schema registries, immutable wrappers or fail-closed validation frameworks without a concrete
  current requirement;
- treat hashes as cryptographic integrity boundaries. They are cache keys, content identities, checkpoint
  compatibility markers or provenance aids, as documented;
- write adversarial tests for unsupported internal states, or use "tampering", "attack", "trust boundary", "fail
  closed" or "integrity violation" for ordinary trusted misuse.

Prefer the simplest implementation that is correct, reliable and fast enough. Accept necessary complexity only when
it is localized, measured, documented and validated (§2.1).

**Defect threshold.** Before calling something a defect and changing code, establish where practical that:
1. a supported or realistically accidental execution path reaches it;
2. a concrete call site, input shape, checkpoint state or user action triggers it;
3. the consequence is observable: wrong scientific output, a wrong branch or selection, a crash, data loss, broken
   documented behaviour, irreproducibility, or material resource waste;
4. a reproducer, failing test, trace or direct code-path evidence supports it.

Classify findings:

| Class | Meaning | Action |
|---|---|---|
| Scientific defect | Can alter scientific conclusions or outputs in supported use | Investigate rigorously |
| Operational defect | A real crash, hang, corruption, lost work or unusable workflow | Make the smallest reliable fix |
| Maintainability issue | Concrete complexity or duplication impeding current work | Address only when in scope |
| Speculative hardening | Needs deliberate misuse, a hostile caller or a hypothetical need, or has no consequence | Do not implement; mention only if useful |

A violated theoretical invariant is not automatically a defect. Never force unsupported states in tests to justify
new production machinery.

### 4.3 Before editing

1. Read every applicable instruction file from the repository root to the target, including nested `AGENTS.md`,
   `AGENTS.override.md`, `CLAUDE.md`, `CLAUDE.local.md` and tool-specific rule files.
2. Read the relevant source and its direct callers.
3. Run `git status --short` and inspect uncommitted changes in the files you may edit. Never overwrite, revert or
   reformat unrelated changes.
4. Identify the expected files and behavioural effect:
   - for non-trivial work, give a short implementation and validation plan;
   - for a purported defect, first meet the defect threshold (§4.2).

### 4.4 Code style

Match the surrounding code.
- Keep scientific logic explicit, with descriptive names for model quantities. Preserve and document units, and
  comment mathematical intent rather than syntax.
- Do not mix broad formatting changes with behavioural ones.
- Use small functions where they clarify the algorithm, but never fragment hot numerical kernels. Keep
  Numba-compatible types and control flow.
- Optimize established bottlenecks without obscuring the science, and avoid unmeasured micro-optimizations. Keep a
  reference or validation path for complex optimized kernels where practical.
- Add no dependency without need and approval. Keep dataset paths out of deep scientific modules.
- Duplication is acceptable when removing it would need a premature framework. Add no wrapper, factory, registry,
  generalized validator or parallel identity scheme for a single speculative case.

### 4.5 Concurrency and multiprocessing

Process-level parallelism and Numba kernels are deliberate; do not change them casually.
- **Pickling and callbacks:** keep the established start method, and keep everything passed to workers picklable.
  **Worker callbacks live at module scope in an importable module, never in `__main__` or inside another
  function.** Forkserver workers cannot unpickle a `__main__` callback (§5.3).
- **Set-up order:** numerical libraries and Numba are configured once at package import (`execution/bootstrap.py`,
  before NumPy and Numba load), and per run by `execution.bootstrap.start_run`. New entry points import through the
  package and use the `execution` pools and CPU budget, never ad hoc pools.
- **Oversubscription:** keep the BLAS, OpenMP, MKL and Numba safeguards. Before combining multiprocessing with a
  threaded library or tool, state the process count, the threads per process and their product. Never nest
  parallelism without a bound on that product.
- **Use the whole allocation:** safely parallel CPU-bound work uses the complete verified CPU affinity (§6.8).
  Widen worker or thread counts when an allocation grows.
- **Before raising concurrency,** account for:
  - per-worker memory, duplicated arrays and deserialized checkpoints;
  - buffers, start-up, serialization and filesystem pressure.

  Choose chunk sizes that avoid excess scheduling, pickling, repeated checkpoint loads and memory duplication.
- **Before changing parallel code,** explain the process and thread model. Consider Numba thread scope, determinism
  and ordering, worker exceptions, and behaviour under Slurm affinity.

### 4.6 Performance

Optimize measured or analytically established bottlenecks, not hypothetical micro-costs, and validate that the
results are unchanged (§2.3).
- **Classify the bottleneck before non-trivial computation:**
  - CPU-bound numerics;
  - compression or decompression;
  - shared-filesystem I/O;
  - memory bandwidth or memory capacity;
  - start-up or serialization;
  - an inherently serial step.
- **Thread counts:** keep light inspection, syntax checks and short tests single-threaded when parallel start-up costs
  more than it saves. Give verified CPU-bound work the whole affinity.
- **Partitioning:** partition by chromosome, contig, region, sample, block, replicate or file where scientifically
  valid. Never parallelize in a way that changes any of these without reporting and validating it: statistical
  dependence, random-number semantics, ordering, convergence or interpretation.
- **File access:**
  - Avoid several workers reading or decompressing the same large shared file.
  - Prefer indexed queries, bounded regions and streaming to full decompression and large intermediate files.
  - Never add concurrency when the measured bottleneck is filesystem throughput or memory bandwidth.
- **Concurrent analyses:** run several expensive analyses at once only when their combined CPU, memory and I/O fit
  the allocation. Keep total processes × threads within the allocation. Go below it only for task-count, memory or
  I/O limits, never to reserve CPUs for orchestration.
- **External tools:** check the documentation and existing usage before assuming `--threads`, `-@` or worker options
  help.
  - For `bcftools`, threads mostly speed compression and decompression.
  - Stream compatible stages through pipes, and prefer BCF or uncompressed BCF streams where supported.
  - Use `bgzip -@` only when compression is the measured bottleneck.
- **Numba:**
  - Never for file I/O, decompression, orchestration or one-off loops.
  - Only for stable, repeatedly executed kernels, after profiling shows Python is the bottleneck.
  - When adding Numba parallelism, verify that it parallelizes, cap its threads, keep a reference path and compare
    the results.
- **Measurements:**
  - Collect elapsed time, CPU, peak memory, I/O, worker counts and exit status from the authorized live run. Run a
    separate benchmark only if the owner asks, or if a wrong resource choice would make the real run unsafe.
  - Treat differences between chromosomes or regions as workload differences.
  - Compare worker counts only when asked, or when changing a shared default.
  - Report when I/O, memory, dependencies or too few tasks prevent saturation.

### 4.7 Pre-run audit

Audit the implementation before a long test, an expensive computation, a production run or a fan-out. The audit
gates the launch; deeper validation gates acceptance.
1. Inspect the final diff and every affected caller, producer, consumer, checkpoint boundary and configuration path.
   Trace the execution path the run will use.
2. Check the code against its biological, statistical and mathematical meaning.
3. Check:
   - boundaries, indexing and interval conventions;
   - eligibility masks, sample order and missing data;
   - output routing, resume identities and completion markers.
4. Check picklability, process × thread limits, shared state, worker error propagation and checkpoint isolation.
5. Run an import or syntax check, then the smallest reproducer or test of the changed behaviour. Where workers are
   involved, use the real execution path (§5.2). Read the warnings and output, not just the exit code.
6. Run `git diff --check`, and confirm that unrelated changes are intact.

Fix every known failure that can affect the run before launching. Never launch expensive work to find errors that an
audit or a reproducer would have found.

Keep the audit proportionate. Broad simulations, scaling studies and benchmarks belong to the authorized run, not to
the gate. Passing it means no launch-blocking defect was found, not that the code is bug-free.

### 4.8 Git

The owner controls history (§1). After editing:
- run `git status --short`, `git diff --stat` and `git diff --check`;
- summarize the diff.

Never stage generated data, checkpoints, logs or results.

### 4.9 Completion report

At the end of a task, report:
1. the files changed;
2. the behaviour changed;
3. the scientific or mathematical assumptions affected;
4. the commands and tests run, with exact outcomes;
5. the tests not run, and the expensive validation still recommended;
6. uncertainties and risks;
7. generated files and jobs;
8. for performance work: allocated CPUs, utilization checks, workers, threads, idle capacity, concurrency changes,
   peak memory and elapsed time;
9. any dependence on unverified biological assumptions;
10. out-of-scope concerns deliberately not pursued.

Do not propose further hardening unless asked.

## 5. Running computations

### 5.1 Running the pipeline

- **Before running:** verify the current entry point and configuration mechanism. Never edit a production
  configuration to test, and never launch the full pipeline unless it was requested.
- **Isolate every attempt.** Give it its own output and checkpoint root, and an explicit code, configuration,
  input, seed and checkpoint identity. Its provisional outputs:
  - publish no global completion marker;
  - overwrite no accepted result;
  - count as accepted only once their validation gate passes.
- **Use a small synthetic or bounded test instead** when asked, when outputs cannot be isolated, or when a failure
  could corrupt accepted results.
- **Resuming from checkpoints:** first confirm the code is compatible with their schema, scientific meaning,
  configuration and cached identities. Weigh compatibility against the cost of recomputation before recommending a
  fresh run. Never recommend deleting checkpoints because the analysis is inconvenient.
- **Deletion:** never delete pre-existing or accepted checkpoints or results without explicit approval. Outputs under
  an authorized attempt's own root may be invalidated as described in §5.2.

### 5.2 Canaries and speculative execution

For authorized work made of scientifically independent units, the pre-run audit (§4.7) gates the launch and deeper
validation gates acceptance.
- **Launch together:**
  - one or more representative canaries, chosen to expose shared failure modes quickly;
  - the remaining units as provisional work, sized to the whole allocation.

  Give canaries the resources to finish first. Never leave CPUs idle waiting for them.
- **A canary uses the real execution path:** the same pool, start method and pickling as the full task, with at
  least two worker processes. An in-process call of the same function does not count. Read its outputs before
  trusting the rest.
- **Speculate only where dependencies are already satisfied.** Never across an unresolved global parameter, shared
  mutable state, a cross-contig dependency, or a stage whose inputs the canary could change.
- **If a canary fails:**
  1. keep the minimum logs needed for diagnosis;
  2. stop the attempt's workers and invalidate its provisional outputs;
  3. fix the cause;
  4. restart under a new attempt identity.
- **Scope of an explicit request to run speculatively:** it authorizes stopping that attempt's workers and deleting
  outputs under its validated attempt root. It never covers pre-existing checkpoints, accepted results or unrelated
  files.
- **Before an expensive launch, state:**
  1. the inputs and attempt root;
  2. the verified allocation and process × thread budget;
  3. the canaries and the decomposition;
  4. the acceptance gate, and which outputs stay provisional;
  5. how workers and outputs are stopped or invalidated if the gate fails;
  6. the expected runtime, memory, I/O and resume behaviour.

  Once the owner has authorized the run, do not ask again, and insert no separate benchmark unless a new material
  risk appears. Collect performance measurements from the live attempt (§4.6).

### 5.3 Every run starts, fails fast and is watched

On 21 October 2026 a delegated replay hung for two hours on a paid node:
- its `__main__`-defined workers could not be unpickled, and the parent process waited forever;
- the agent had ended its turn, and the coordinator watched only exit codes.

- **Confirm every start** (owner, 21 October 2026). This applies to any launch, by you or a delegated agent: a local
  or background process, a pool task, an `srun` step or an `sbatch` job.
  - Within about five minutes, check that it really runs:
    - its processes are alive on the expected node and using CPU;
    - its log shows the expected start-up lines and keeps growing;
    - set-up (imports, kernel warm-up, pool start) has finished, and the first unit has begun.
  - A successful submit, a PID, a claim file or `Submitted batch job` shows only that a launch was attempted.
  - Check again when the first unit or stage should have finished. Then check at the run's stated cadence until it
    ends, recording each check in the progress file.
  - Do not end a turn straight after a launch. Wait, sleeping rather than polling, until the start check passes,
    and leave the next check scheduled.
  - A pending Slurm job has not started. Follow its expected start time and pending reason within the polling
    limits (§6.6), then use its logs and heartbeats.
  - If a run has exited, failed or stopped progressing, read its log at once, and fix or report the cause.
- **Fail fast; never hang.** A pool or background task exits non-zero within seconds when any worker raises, dies or
  cannot start.
  - Propagate worker exceptions, for example through `concurrent.futures` or the package's pools, which re-raise.
  - Give every wait a timeout or a liveness check.
  - Keep callbacks importable (§4.5).
- **Report progress.** A task expected to run for more than about 15 minutes writes a progress or heartbeat file at
  least every few minutes, and its log keeps growing. Whoever submits it states the expected duration and the
  progress signal.
- **Judge liveness by progress, not by claims.**
  - At every periodic check, compare each claimed task's newest log or progress write with its cadence.
  - Inspect on its node any task silent for about 20 minutes, or longer than its stated heartbeat: process states,
    CPU use and the log tail.
  - Near-zero CPU and no writes means the task is hung:
    1. stop it through its verified process tree (§7.2);
    2. keep its log and tell its owner;
    3. requeue it only after the cause is fixed.
  - A stall on a paid allocation is urgent. Release that allocation if no other useful work remains.
- **Hand over running work explicitly.**
  - An agent that ends its turn while a task runs records the task, its expected completion time and its progress
    signal.
  - The coordinator resumes it on completion, failure or stall, and applies the start check to every run a delegated
    agent reports.
  - "No error code" is never evidence that a task is running.

### 5.4 Keep allocations busy with useful work

The owner pays for every core of a paid allocation, the hosting node included. On 21 October 2026 they asked that
"the nodes we have allocated to us aren't being left idle or underutilised".
- **Every 30 minutes during a campaign,** the coordinator checks every allocation: the hosting node, the pool workers
  and any others.
  - One cheap pass covers heartbeats, claimed and pending tasks, and host load, plus at most one `squeue`.
  - Record it as one status line, and reuse the same pass for the stall checks (§5.3).
- **Also reassess utilization:**
  - when a pool starts and when a canary finishes;
  - at stage, batch or chromosome boundaries;
  - when workers exit or a straggler tail begins;
  - whenever you regain control during a long run.

  Measure against the verified affinity (§6.8), not `nproc`, and record:
  - allocated CPUs, active workers and threads per worker;
  - useful occupancy and idle capacity;
  - any adjustment made.
- **Idle or underused** means a live worker with nothing claimed, a node whose load stays well below its cores while
  it holds a task, or an allocation with no task.
  - First check whether dependencies, memory capacity or bandwidth, filesystem I/O or a serial step explains the
    idle CPUs.
  - Then act, in this order:
    1. queue the next useful authorized work: critical-path tasks first, then validation that can still change a
       decision;
    2. widen running work, so that freed cores pass to unfinished tasks;
    3. let bounded agent tests use a shared host's idle cores;
    4. release non-SL4 allocations under §6.7.
  - Record the cause and the action. If more concurrency would not help, keep the allocation as it is and state the
    limiting factor.
- **Reallocate as tasks finish;** never keep an early, conservative worker count. Where the validated path supports
  it, schedule dependency-ready batches rather than whole chromosomes. Keep batch boundaries, stable ordering,
  single-writer checkpoints and resource limits. `docs/performance.md` documents where each stage stands.
- **Order pool tasks** so that critical work is claimed first, and keep a short ready queue for nodes about to start.
- **Before a session or turn ends with follow-on work queued,** make sure workers whose walltime covers that work will
  exist while nobody is watching. Otherwise run it on the hosting node through a durable waiter.
- **Never fill nodes with busywork** (§5.5), and never put whole-genome work on a partial node (§6.5).

### 5.5 Proportionate validation and superseded work

Compute is limited. The owner, 19 October 2026: "we aren't made of compute you know. Please don't go overboard on
validations and tests and runs".
- **Ask whether a run can change a decision.** If no plausible outcome would change what happens next, do not launch
  it.
- **Check the queue first.** Never queue a test whose answer, or a newer version of it, is already queued or known.
- **Validate proportionately.**
  - Use focused, bounded tests for what changed.
  - Run broad matrices (every dataset, noise cell, reversal and renaming) once, on the final integrated candidate.
  - Never rerun checks whose code and inputs have not changed.
- **Cancel superseded work at once.** A task is superseded when its question has been answered, or when a newer
  candidate, tree or configuration replaces what it tests. Examples:
  - the noise matrices of a rejected candidate;
  - the timings of a replaced tree;
  - duplicates of a running check.
- **A change that forces revalidation cancels the old validation in the same step** (owner, 21 October 2026), so
  that its nodes go to the new tasks:
  - remove or hold the old pending and held tasks;
  - stop the old running tasks through their verified process trees, checking for orphans;
  - cancel pending Slurm requests that served only the old work;
  - give the freed allocations the new tasks. Running SL4 workers stay until they expire.

  Keep completed outputs and the logs needed for diagnosis, and record what was cancelled and why.
- **A change that cannot alter any computed result does not supersede validation of the same science.** An example
  is an operational fix that leaves every output identical by construction. Say so when keeping such runs.
- **In a shared queue,** the task's owner asks the coordinator, who acts.
- **Do not busy-wait.** A waiting agent ends its turn, or sleeps without re-reading large state. Every poll costs
  tokens as well as scheduler or filesystem load.

### 5.6 Durable state and interruptions

Usage limits can cut an agent off, and allocations can end abruptly.
- **Storage:** keep important work on shared project or home storage. Node-local scratch is only for reproducible,
  disposable data.
- **Atomic writes:** write outputs atomically where practical: write a temporary file, validate it, then rename it.
- **Agent records:**
  - Keep a progress file with each step, its outcome and a `NEXT:` line.
  - Check for an existing result before redoing work, and write results atomically before returning.
  - A cut-off agent is interrupted, not passed or failed.
  - Coordinators keep the plan, the handoff notes and the status log on project storage.
- **After an interruption:**
  - inspect `git status --short`, `git diff --stat` and `git diff --check`, changed and untracked files, and known
    output locations;
  - classify each output as complete, partial, corrupt, absent or uncertain before overwriting it;
  - do not rerun expensive work until the last confirmed completed step is known.

## 6. CSD3 and Slurm

### 6.1 The protected hosting allocation

**Never release, cancel, shorten or resize an allocation hosting Codex, Claude Code or their runners, and never
attach a cleanup timer to it.**
- This binds every agent. It overrides completion cleanup, cost optimization and the idle limits.
- The allocation's scheduler expiry still applies. This rule does not authorize extending it or changing the runner
  set-up.
- Before releasing any allocation, identify and record the hosting job, using the runner host, the Slurm job and
  cgroup information, and the job list.
  - An unset `SLURM_JOB_ID` does not prove the runner is outside an allocation.
  - If ownership is uncertain, leave every allocation that might host a runner untouched.
- Killing what keeps an allocation alive ends it exactly as `scancel` does. Never signal any of these (§7.2): its
  `screen` or `tmux` session, its login shell, `sshd`, `slurmstepd`, `systemd`, or the runner itself.
- Keep the runner separate from worker allocations where the set-up permits. Never change the runner or the sandbox
  just to acquire resources. A written handoff never permits releasing the hosting allocation.

### 6.2 What agents may request and manage

**What the owner authorizes.** Agents on authorized tasks may, without asking each time:
- inspect resources and balances;
- submit SL4 allocations and run work in them;
- resize, release, cancel or replace the allocations they manage.

This covers neither unrelated scientific work, nor a full pipeline run that was not requested, nor anyone else's
jobs.

**Service levels:**
- **SL4-CPU only.** The owner, 17 October 2026: "Don't request SL3 nodes unless I tell you to ... request SL4 nodes
  only, they will come online eventually".
  - Request SL3-CPU, SL2-CPU or SL3-GPU only when the owner asks at the time, and only for that request.
  - When SL4 requests wait on the shared budget, leave them pending, and say how long the work may slip.
  - Never use SL2-GPU or any other GPU service level.
- The accounts are normally DURBIN-SL4-CPU, DURBIN-SL3-CPU, DURBIN-SL2-CPU and DURBIN-SL3-GPU. Verify access rather
  than assuming it.

**Before requesting more:**
- Reuse suitable capacity already assigned to the task, coordinating with the tasks that share it.
- Optimize useful progress per credit and per hour. A window too short to finish or checkpoint the work saves
  nothing.
- Inspect, reusing recent shared observations rather than re-querying for each job:
  - the user's jobs and `hpcstat` (or `sinfo`);
  - the relevant `scontrol` partition, job and QoS records;
  - balances (`mybalance`).

  Idle nodes alone do not prove that a request can start.

### 6.3 Limits

- **At most 448 allocated CPUs at once across SL3-CPU and SL2-CPU combined.** Count all of the user's jobs, using
  actual CPU counts, including memory-driven extras:
  - manual jobs and retained idle allocations;
  - the hosting allocation, if it is on SL3 or SL2;
  - pending requests that could start together.

  At the cap, reuse assigned capacity or defer. Never cancel unrelated jobs, or the hosting allocation, to make room.
- **SL4-CPU, and host CPUs allocated through GPU partitions, are outside that cap.**
  - SL4 has no user-imposed core cap, but every Slurm limit applies: account, association, QoS, partition, resource
    and walltime.
  - That includes the shared `GrpTRESRunMins` budget: cores × remaining walltime, with periodically refreshed
    displayed usage.
  - Read the current limits. Never treat an observed amount of free SL4 capacity as a quota.
  - GPU host CPUs must fit the actual GPU allocation and the site limits. Never infer them from node size.

### 6.4 SL4 workers

- **Check `hpcstat` before choosing a partition** (owner, 19 October 2026), at each submission decision.
  - Never queue a partition with no free capacity when another suitable one has idle nodes. A higher tier does not
    create free nodes.
  - Cancel pending requests on a partition that has filled up while suitable capacity is idle elsewhere.
- **Partition preference:**
  - Prefer `icelake-himem` for suitable Ice Lake requests, especially workers of half a node or more. It has the
    higher scheduling tier (§6.9), and Slurm considers tier before job priority.
  - Fall back to `icelake` when high-memory nodes are inaccessible or unsuitable, or when admission evidence favours
    it.
  - Fall back to `sapphire` when it has idle nodes and the Ice Lake partitions do not.
  - Record the reason for any fallback.
  - Small workers may prefer `sapphire` when it starts promptly and its per-core speed helps. When admission is
    constrained, prefer `icelake-himem`, then `icelake`.
- **Shape and walltime:**
  - Request genuine CPU and memory needs, never inflated memory, and count memory-driven extra CPUs in the ledger.
  - The default large worker is 48 CPUs or a full 76-CPU node, for two hours. Longer walltimes are for queued work
    that needs them: for example, the 12-hour full nodes the owner requests for validation campaigns.
  - Shorter requests can start sooner, provided the work can finish or checkpoint within them. Use checkpoint and
    resume for longer investigations.
  - Never understate runtime to gain admission, and never assume an extension.
- **Build capacity incrementally.** The shared budget admits small commitments as headroom opens, while a large
  request waits. For substantial checkpointable work:
  - keep a bounded queue of genuinely useful workers, sized to the remaining independent units;
  - serve them through a shared atomic task queue, so that workers starting at different times claim distinct units;
  - let the scheduler admit them.

  Never race submissions, churn jobs, or send admission-only probes or duplicate requests to compare partitions.
- **Keep running SL4 allocations until they expire.** The owner, 7 October 2026: "don't free up SL4 until they
  naturally expire, I don't get charged for those".
  - Never release, cancel or shorten a running SL4 allocation, whether by hand or through a cleanup timer, a reduced
    TimeLimit, a completion hook or a scheduled cancellation.
  - Give idle SL4 workers useful authorized work instead, and preserve their checkpoints across normal expiry.
  - This overrides the release and idle rules of §6.7, for running SL4 allocations only.
  - Pending SL4 requests follow the pending-request rules. Pacing, sizing and the shared budget still apply.
  - Never submit SL4 workers that no authorized work can use.
- **Pending requests:** bound their lifetime with a realistic deadline or a durable cancellation. A completion
  deadline must allow for both the wait and the runtime. Cancel surplus pending workers after counting retained
  allocations, and both ready and known upcoming work.
- **Records:** record every job's account, requested and actual cores and memory, submission, start, expiry, task
  ownership, checkpoint state and release action.
  - Track useful task time separately from allocated CPU-hours.
  - Recover an interrupted task claim only after its former writer is confirmed terminal.
  - Preserve healthy running work and checkpoints when changing future requests.

### 6.5 Whole-genome runs need full nodes

**Never run a whole-genome workload on less than one full node's allocated CPUs** (owner, 4 October 2026).
- **Scope:** production, validation, merge-correctness checks, resume tests and canaries. That includes "the last
  remaining check", and a resume expected to be short. It applies to delegated agents and checkers too.
- **Default node:** a full 76-CPU Ice Lake node, reused across successive whole-genome tests while its walltime
  suffices, and requested on SL4 (§6.2). For other node types, use their verified full-node CPU count.
- **Before launch:** record the job ID, allocated CPUs, memory, remaining walltime and worker affinity. Verify that
  the full node is allocated to this run: not merely its host, and not shared with other tasks.
- **Use all of it:**
  - pass the full CPU budget through the launcher, harness, checker and scientific workers;
  - set process × thread limits that prevent oversubscription;
  - leave no hard-coded 16-, 48- or other partial-node cap.

  Explain any genuine serial, memory or I/O limit, rather than silently using less.
- **No fallback:** if no full node is available, queue, coordinate or defer. Never fall back to a partial
  allocation.
- **Bounded checks:** genuinely bounded unit, fixture or single-chromosome checks may use smaller allocations.
  Calling a whole-genome run a test does not make it bounded.

### 6.6 Scheduler traffic

Every scheduler call costs controller work, not just polls: submissions, step launches, cancellations, updates,
accounting queries and client retries. A separate process, node, account or service level is no separate
allowance.
- **Poll `squeue` at most once every 120 seconds,** across all agent checks and scripts.
  - Query all relevant jobs in one call, and share the timestamped snapshot.
  - Poll less often when no decision depends on it.
  - Never substitute `sacct`, `scontrol` or other status calls to see the same state sooner.
  - Make focused job, accounting and resource checks at decision points, batched.
- **Aim for about one controller request every two minutes in steady operation,** across coordinated tasks.
  - Start-up, recovery and cleanup may burst above it. Keep bursts bounded, record why, and amortize them over
    useful work.
  - A shell command is not one RPC: count its submission, step start and completion traffic.
  - Reuse allocations and long-lived workers, and distribute units through task queues, with no Slurm call per unit.
  - Check the site's rate-limit settings at set-up (§6.9) and reuse them. Never use client backoff as the pacer, and
    never probe the remaining allowance.
- **Grow allocations through one submission opportunity every five minutes,** shared across cooperating tasks, and
  only when fresh demand justifies it.
  - Prefer one small, demand-sized job array per resource profile to separate jobs. Different profiles take separate
    opportunities.
  - Before each submission, refresh demand from cheap workflow state, and count running, pending and in-flight
    capacity.
  - Bound the submissions per cycle; a sleep after an unbounded loop is not pacing. Stop when capacity covers useful
    demand.
  - This traffic budget is not an SL4 core cap.
- **Arrays:**
  - Before a coordinator uses them, make its ledger, queue parsing, recovery and cleanup array-aware.
  - Record array and task identities, and count every element that could run, pending and in-flight included,
    against the limits. Keep per-element ownership and logs.
  - Array elements also have their own raw job IDs, which logs and worker names may use.
  - Site array and job limits, the paid-CPU cap and the QoS limits all still apply.
- **One coordinator** runs routine campaign queries and cleanup, sharing snapshots and a launch ledger.
  - Group queries and cancellations.
  - For bulk whole-job cancellation, prefer `scancel --ctld` with explicit owned selectors and server-side state
    filters such as `--state=PENDING`, excluding the hosting allocation.
  - Use filesystem heartbeats and completion records for ordinary progress.
- **If calls back off or slow down,** pause growth while work continues. Never add rapid retries or replacement
  queries. Validate changes with sparse diagnostics, not a new poller. `sdiag` counts omit some rejected attempts,
  and its handling time is neither CPU time nor harm to other users.
- **Keep scheduler and bookkeeping errors separate from scientific failure.**
  - Pause submissions and record the error, while healthy workers continue.
  - Never broadcast a scientific failure, or cancel a campaign, over a coordinator exception.
  - Recover claims only once their former writers are confirmed terminal.
- **Coordinator changes:**
  - Before replacing a coordinator, or retrying an ambiguous submission, reconcile any in-flight submission and its
    durable record.
  - Never hold a task-claim lock across a blocking Slurm call, and keep capacity reservation and ledger updates
    atomic.
  - Editing documentation does not change a coordinator that is already running.

### 6.7 Reuse, release and idle limits (non-SL4 allocations)

None of this applies to running SL4 allocations (§6.4) or to the hosting allocation (§6.1).
- **Reuse.**
  - For a sequence of tests or development runs, obtain one suitably sized and long allocation, and run the
    sequence in it with headroom and checkpoints.
  - Keep a managed allocation through task boundaries and dependency gaps when known, compatible follow-on work will
    use its remaining CPUs, memory and walltime (owner, 4 October 2026). For example, a two-hour worker that
    finishes its first task in 12 minutes keeps the remaining 1 h 48 min for the next tasks.
  - Record the next use, its expected readiness and the allocation's expiry. Each next task must fit, or checkpoint,
    in the remaining time.
  - Reuse needs no evidence of contention.
- **Routing and records.**
  - Route work into an allocation explicitly, through `srun --jobid=...` or the allocated-node session; having an
    allocation does not move your shell there.
  - Record each managed job's ID, account, resources, purpose, sharing and expiry, so that another agent can reuse or
    release it without guessing.
- **Release.**
  - Release a managed allocation promptly once its authorized and known follow-on work is exhausted, and cancel
    pending requests that are no longer needed.
  - Beyond planned reuse, retain one temporarily only when current scheduler evidence shows that a replacement would
    be hard to get soon.
  - Record the evidence, expected reuse, cost, job ID and release deadline, and reassess when the work or the owner
    returns.
- **Two-hour idle limit.** An idle allocation kept while awaiting the owner is released after two hours without owner
  input in the task using it, or at its expiry if sooner.
  - If two hours have already passed when the known work ends, release it at once.
  - Agent messages, polling and heartbeats do not reset the clock.
  - The limit does not apply to an actively managed workflow and its known follow-on work.
  - Coordinate shared allocations with their other tasks.
- **Durable cleanup.** Before leaving an idle allocation, arrange cleanup that survives your turn ending:
  - a scheduler-enforced expiry (a reduced TimeLimit counts from job start, so set the elapsed time plus the allowed
    retention), or
  - a durable job-specific timer, replaced or disarmed before useful work resumes.

  You may not be able to extend a job again. If reliable cleanup cannot be arranged, release the allocation before
  ending the turn.

### 6.8 Where you are and what you have

- **Where you are.** You may be in an interactive `sintr` allocation, or on an allocated node reached by SSH;
  determine which. Never run substantial computation on a login node. Light inspection, syntax checks and small
  tests are fine where permitted.
- **Find the allocation with:**
  - `hostname`;
  - the `SLURM_*` variables: `JOB_ID`, `CPUS_ON_NODE`, `CPUS_PER_TASK`, `JOB_CPUS_PER_NODE`, `MEM_PER_NODE` and
    `MEM_PER_CPU`;
  - `grep Cpus_allowed_list /proc/self/status` and `os.sched_getaffinity(0)`;
  - `squeue -u "$USER"`.

  Missing Slurm variables do not prove there is no allocation. Match the hostname, ask when several jobs could match,
  and read `scontrol show job JOB_ID` for the authoritative resources.
- **Size work by** the verified affinity, the job record and the cgroup cpuset. `nproc` can honour `OMP_NUM_THREADS`
  and under-report. Never infer the allocation from node size, `/proc/cpuinfo` or `nproc --all`.
- **Use all allocated resources, and only those.** Reserve no CPUs for the agent, the OS or orchestration. Use
  reviewed `srun` commands where required, and reviewed `sbatch` scripts for long or production work.
- **`hpcstat`** is defined in the login profile. From a non-login shell, run `bash -lc hpcstat`.

### 6.9 Dated CSD3 observations

These were true when recorded. Verify them at set-up, or when behaviour changes, not for every job.

| Date | Observation |
|---|---|
| 27 Sep 2026 | The controller rate limit was a 40-token bucket, refilled by eight tokens a minute. This is not a target rate. |
| 27 Sep 2026 | Staggered, mostly 48-CPU, two-hour `icelake-himem` workers reached 688 running SL4 CPUs across 14 jobs, with 96 more pending. An example, not an entitlement. |
| 29 Sep 2026 | `icelake-himem` had `PriorityTier=2`, and `icelake` and `sapphire` had tier 1. Within a tier, an older array's age-based priority can put it ahead of a new request ([Slurm scheduling](https://slurm.schedmd.com/priority_multifactor.html)). |
| 21 Oct 2026 | A full node had 76 CPUs on `icelake` and `icelake-himem`, and 112 on `sapphire`. |

## 7. Shell, processes and data

### 7.1 Commands

- **Explain** non-trivial commands before running them.
- **Get explicit approval** before:
  - installing or upgrading software, or changing the Conda environment;
  - changing Git history;
  - overwriting accepted results;
  - deleting or moving files outside your attempt root, or recursively scanning large directories;
  - launching a full pipeline that the current request did not authorize.

  Allocation management within §6.2 needs no further approval. Other people's jobs are never yours to touch.
- **An explicit request to launch a run approves that run;** do not ask again.
- **Avoid destructive commands:** `rm -rf`, `git clean`, `git reset --hard` and broad wildcard deletion. Prefer
  bounded output: `tail -n 100`, `grep -n -C 5` and `sed -n 'A,Bp'`.
- **Secrets:** never expose secrets, credentials, tokens, private keys or unrelated environment variables, and never
  print a whole `.env` file.

### 7.2 Stopping processes safely

On 15 October 2026 a cleanup loop over `ps -u "$USER" -o pid= --ppid PID` signalled every process the user owned on
the hosting node, because procps combines selection options as a union. It killed the `screen` session that kept the
hosting job alive, and so ended every agent.
- **Stop only processes your task started.** Record their PIDs, or a dedicated process group, when you start them.
- **Never choose targets through a user-wide selection:** no `ps -u`, `pgrep -u` or `pkill -u`.
  - Never use `killall`, `kill -1` or `pkill`. `pkill -f` also matches its own shell, and any unrelated command line
    containing the pattern.
  - Never combine `ps` selection options expecting an intersection.
- **Process groups:** use `kill 0` or `kill -- -PGID` only for a process group that your task created itself (for
  example with `setsid`), and whose ID you have checked.
- **Stopping a tree:** start from a verified PID, and walk its descendants through parent links (`pgrep -P`
  recursively, or `/proc/PID/stat`), excluding your own shell and its ancestors. Afterwards, check for orphaned
  children.
- **Never signal session infrastructure:** `screen`, `tmux`, `sshd`, `slurmstepd`, `systemd`, an allocation's login
  shell, or the Codex or Claude runner (§6.1).
- **Inspect computed target lists.** Print any computed target list first. If it holds anything but your task's
  processes, send nothing.
- **Briefs:** put these rules in the brief of every delegated agent that may start or stop processes.

### 7.3 Data and generated files

The data are cichlid fish, not human participants, but the files are large.
- **Inspect the minimum:** headers, indexes, sizes, counts, metadata and bounded samples.
- **Never:**
  - print thousands of records;
  - recursively scan large trees without approval;
  - copy large datasets into the repository;
  - invent paths.
- **Summarize large results locally.** Never expose unnecessary records, sample identifiers, logs or bulk output to
  external model context.
- **Never commit datasets, checkpoints, logs or results:**
  - `*.pkl.b2`, `*.p5.b2`, `.pipeline_checkpoints*`, `results_*` and `logs/`;
  - VCF, BCF, BAM and CRAM files;
  - validation CSVs and run summaries.

  Check `.gitignore` and `git status` before staging. Never delete checkpoint or result directories without explicit
  approval.

### 7.4 Sandbox compatibility on CSD3

**The namespace error.** An agent may already run inside a user namespace, so a tool that adds another Bubblewrap
layer can fail before running with:

```text
bwrap: Creating new namespace failed: nesting depth or /proc/sys/user/max_*_namespaces exceeded (ENOSPC)
```

This is a namespace failure, not a full disk.
- Run ordinary commands through the orchestrator's normal workspace sandbox (in Codex, possibly `csd3-workspace`).
  Never request full access or escalation just to avoid it.
- Edit with a reviewed unified diff and system `patch`, unless a higher-priority instruction requires a patch
  helper. Inspect the result, and stay inside approved roots.
- If a required helper fails once with this error, use the permitted fallback rather than retrying.
- Validate as usual, including `git diff --check`.
- If ordinary commands fail the same way, stop and report it. Never bypass access controls; this file cannot
  disable an outer sandbox.

**The Codex runner's temporary directory** is persistent private storage, set host-side in `/home/ahk39/.codex/.env`
as `TMPDIR=/rds/user/ahk39/hpc-work/codex-runner-tmp.gKhMfcvc`.
- Preserve the setting and the directory during cleanup. It is runner scratch, not a place for results or
  checkpoints.
- After a runner restart or host change, check `hostname`, `TMPDIR`, and `df -h` and `df -i` of the directory before
  substantial work.
- A distinct error was cured on 16 September 2026 by persisting this setting and restarting the runner:

  ```text
  failed to register synthetic bubblewrap mount target /tmp/.git: No space left on device (os error 28)
  ```

  Do not mistake it for the namespace error, or conclude that the project filesystem is full. Check the runner host,
  its effective temporary directory and its quotas. A separate login terminal may differ.
- Changing the start-up setting needs explicit authorization and a runner restart.
  - Never purge live sandbox state, kill unrelated runners or jobs, or add automatic checkpoint deletion.
  - Inspect only the `TMPDIR` line of `.env` (§7.1).

## 8. Delegation and orchestration

Use several agents when that improves reasoning, implementation, validation or elapsed time. No concurrency or depth
cap is set here, beyond the resource rules.
- **Decompose** around concrete biological, mathematical, implementation, code-path or validation questions. Roles
  such as scientific-modeler, code-path-explorer, scientific-validator and pragmatic-implementer help.
- **Every brief carries** the task contract (§4.1), the scientific assumptions, the paths, the acceptance criteria and
  the resource limits. It also carries the rules agents most often break:
  - safe process stopping (§7.2);
  - no AulStu tests (§3.1);
  - SL4 only (§6.2);
  - no Git history operations (§4.8);
  - start checks and fail-fast tasks (§5.3);
  - cancelling superseded validation (§5.5).
- **Delegation changes neither the scope nor the threat model.** Agents' proposals still meet the defect threshold
  and the validation rules.
- **The coordinator** synthesizes the results, reconciles conflicting assumptions, separates evidence from inference,
  and stays accountable for the implementation and the report.
- **Agents return durable outputs:** edits, test artifacts, findings, assumptions, uncertainty and the exact next
  action. They keep progress files (§5.6). Unreported reasoning is never the only record.
