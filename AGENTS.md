# AGENTS.md

Instructions for AI coding agents and orchestrators working in this repository.

Read this file before inspecting, modifying, or executing the project. This is the canonical shared repository guidance across coding tools; tool-specific adapters may import or supplement it but must not contradict it. Follow the current repository code, data documentation, and explicit user instructions when they conflict with assumptions here.

## Standing authorization for HPC allocation management

The user delegates efficient CSD3 allocation management to Codex, Claude, and
other agents working on authorized project tasks. This is standing approval to
inspect resources and balances, submit allocations, run work within them, and
resize, release, cancel, or replace the allocations managed for those tasks.
Do not ask for approval for each allocation or service-level fallback covered
below. This includes SL2-CPU spending when the cheaper permitted options are
unsuitable for the authorized work. It does not authorize unrelated scientific
work, a full pipeline run that was not requested, or changes to unrelated jobs.

### Protect the allocation hosting Codex

**Never release or cancel the allocation on which Codex or its remote runner
is running.** Do not shorten its walltime, resize away its host resources, or
attach an automatic cleanup timer that could terminate it. This prohibition
applies to every agent, including Claude, and overrides completion cleanup,
resource-cost optimization, and the two-hour inactivity rule below.

Before releasing an allocation, identify and record the protected hosting job
using the runner host, Slurm job/cgroup information, and the user's job list.
An unset `SLURM_JOB_ID` does not prove the runner is outside an allocation.
If ownership of the runner host is uncertain, leave any potentially hosting
allocation untouched until it is identified. Its existing scheduler expiry
still applies; this rule does not authorize extending it or changing the
runner setup.

### Account and partition selection

- Reuse suitable available resources in an allocation already assigned to the
  task before requesting more. Coordinate with other tasks sharing it.
- Prefer suitable available GPU-partition capacity using **SL3-GPU only** over
  a new CPU-partition allocation. The user wants to use the otherwise wasted
  GPU allowance, including its allocated host CPUs for CPU work where site
  rules permit. Do not require a GPU port of the scientific code to consider
  this option. Never use SL2-GPU or another GPU service level.
- For CPU partitions, always consider the cheapest permitted service level
  first: **SL4-CPU -> SL3-CPU -> SL2-CPU**. Use the actual accessible project
  accounts (normally DURBIN-SL4-CPU, DURBIN-SL3-CPU, DURBIN-SL2-CPU, and
  DURBIN-SL3-GPU); verify their availability rather than assuming access.
- Fall back autonomously when limits, balances, queue conditions, memory,
  runtime, or hardware/software compatibility make the preferred option
  unsuitable. Record why the cheaper option was skipped. Do not wait
  indefinitely or submit duplicate live allocations just to try every tier.
- Optimize useful progress per credit and elapsed time. Avoid unnecessary
  paid CPU time and idle GPU reservations; SL4 preference is not a reason to
  choose a window too short to finish or checkpoint useful work.

### Resource selection and aggregate limits

Before submission, inspect the user's existing jobs, `hpcstat` or `sinfo`,
relevant `scontrol` partition/job/QoS records, and balances such as `mybalance`
when available. Include memory-driven extra CPU allocations and CPU/GPU ratios
in the estimate. Idle nodes alone do not prove that a request can start.
`GrpTRESRunMins` is a shared cores-times-remaining-walltime limit; its displayed
usage is periodically refreshed. Read current limits rather than hardcoding
an observed amount of free SL4 capacity or treating it as a personal quota.

- **Never exceed 448 allocated CPUs at once across SL3-CPU and SL2-CPU
  combined. SL4-CPU allocations do not count toward this cap.** Sum all of
  this user's SL3/SL2 CPU accounts, allocations, and agent tasks, including
  manually started jobs, retained idle allocations, and the protected Codex
  hosting allocation if it uses SL3-CPU or SL2-CPU. This user-imposed cap is
  aggregate, not a separate allowance for each tier, job, or partition.
- Include pending SL3/SL2 CPU requests that could start together when checking
  that cap; coordinate concurrent agents' submissions so they cannot overbook
  it. Use actual allocated/requested CPU counts, including extra CPUs required
  for memory. At the cap, reuse assigned capacity or defer new SL3/SL2 CPU
  allocations; SL4 work may still be requested. Do not cancel unrelated jobs
  or the protected hosting allocation to make room.
- **SL4-CPU has no additional user-imposed core-count cap.** Size SL4 workers
  for useful authorized work independently of the 448-core SL3/SL2 allowance.
  This does not make SL4 unlimited: obey current Slurm account, association,
  QoS, partition, resource, and walltime limits, including the shared
  `GrpTRESRunMins` budget. Verify these live rather than assuming that the
  absence of a per-user CPU limit guarantees admission. The existing rules
  for efficient use, allocation ownership, and cleanup still apply to SL4.
- Host CPUs allocated through GPU partitions are also outside the combined
  SL3/SL2 CPU cap, but must fit the actual GPU allocation and all current
  site/account CPU, memory, GPU-count, and runtime limits. Do not infer usable host CPUs
  from the physical node size or assume every GPU allocation grants a node.
- **Poll `squeue` no more often than about once every two minutes (120
  seconds).** This applies to agent-issued checks and monitoring scripts, not
  just explicit polling loops. Query all relevant jobs together and reuse the
  timestamped result between checks; do not issue separate polls per job or
  output format. Coordinate tasks sharing allocations so they can reuse a
  recent queue snapshot rather than each polling independently.
- Two minutes is a minimum polling interval, not a requirement to poll that
  often. Prefer slower or event-driven checks when no scheduling decision is
  needed. Do not substitute repeated `sacct`, `scontrol`, or other Slurm
  status calls merely to obtain the same queue state more frequently.
- Use focused job/accounting/resource checks at useful decision points and
  batch them where possible. Do not tightly poll Slurm, repeatedly submit
  probes, or churn queued jobs merely because a resource snapshot changes.

### Building useful SL4 capacity incrementally

SL4's shared `GrpTRESRunMins` limit is an outstanding CPU-times-remaining-walltime
budget, not a fixed personal core quota. A large request can remain blocked
while smaller commitments start as headroom becomes available. In the
27 September 2026 real-cross campaign, staggered, mostly 48-CPU/two-hour
Ice Lake high-memory workers grew to 688 running SL4 CPUs across 14 jobs,
with another 96 CPUs pending at the 10:12 BST snapshot. This is a historical
example, not a guaranteed capacity entitlement or a required worker shape.

For substantial authorized work with independent checkpointable units:

- Maintain a bounded queue of genuinely useful SL4 workers, rather than
  requiring enough instantaneous headroom for the entire desired core count.
  Size the queue to remaining independent work; do not reserve workers with
  nothing useful to claim.
- Choose realistic core-by-walltime commitments. For example, 48 CPUs for
  two hours initially needs 5,760 CPU-minutes. Shorter requests can admit
  sooner than whole-node, twelve-hour reservations, provided the useful work
  can finish or checkpoint within that window. Do not understate runtime
  merely to gain admission, assume an extension, or bypass scheduler limits.
- Use a shared, atomic task queue so workers starting at different times
  claim distinct chromosomes or other independent units. Reuse each worker
  for further eligible tasks, then exit immediately when none remain.
  Preserve intermediate checkpoints across normal scheduler expiry.
- Consider suitable accessible partitions, including Ice Lake high-memory,
  using current partition rules, memory needs and actual admission evidence.
  Prior success on one partition is not a guarantee; requested memory must
  not silently increase allocated CPUs beyond the ledger.
- Let the scheduler admit queued requests as capacity opens. Reassess at
  useful task boundaries, respecting the existing minimum 120-second queue
  polling interval. Do not race submissions, churn jobs, or run repeated
  admission-only probes when useful queued work can provide the evidence.
- Bound pending lifetime with a realistic deadline or durable cancellation.
  A Slurm completion deadline must allow both pending wait and requested
  runtime. Cancel unused pending workers when the useful backlog disappears.
- Record every job's account, requested and actual cores/memory, submission,
  start, expiry/end, task ownership, checkpoint state and release action.
  Recover an interrupted task claim only after its former writer is confirmed
  terminal. Track useful task time separately from full allocated CPU-hours.
- SL4 remains outside the user's combined 448-core SL3/SL2 cap, but all site
  limits still apply. Use permitted SL3 capacity for the critical path when
  SL4 admission would delay the task; never disturb the protected runner.

### Allocation reuse and execution

For a long sequence of tests or development runs, obtain one appropriately
sized, sufficiently long allocation and run the sequence within it instead of
requesting a fresh allocation for every test. Estimate the whole sequence's
runtime and memory, allow practical headroom, and use checkpoint/resume when
needed. Keep process-by-thread use within verified affinity; the existing
pre-run audit and scientific validation requirements still apply.

Record each managed job's ID, account, resources, purpose, ownership/sharing,
and expiry so another agent can reuse or release it without guessing. An
allocation being available does not move the agent's shell there: route work
explicitly through `srun --jobid=...` or the supported allocated-node session.
Keep the runner's lifetime separate from worker allocations where the existing
setup permits. Preserve the allocation hosting Codex under the protection
rule above; a written handoff is not permission to release it.
Do not change the runner setup or sandbox merely to acquire resources.

### Release, overload retention, and two-hour inactivity limit

These cleanup and retention rules apply only to worker allocations that do
not host Codex or its remote runner. The protected hosting allocation must
never be selected for release or automatic inactivity cleanup.

- Release managed allocations promptly when the authorized work is done, and
  cancel pending requests that are no longer needed.
- Exception: retain a useful allocation temporarily if current scheduler
  evidence shows substantially worse contention and obtaining a replacement
  soon would be difficult. Record the evidence, expected reuse, cost, job ID,
  and release deadline. Do not hold resources merely because they were hard
  to acquire earlier, and reassess retention when work or user input resumes.
- An idle allocation retained while awaiting the user must be released after
  **two hours without user input** in the task using it, or at its existing
  expiry if sooner. If two hours have already elapsed when useful work ends,
  release it immediately. Agent messages, polling, and automated heartbeats
  do not reset this clock. Coordinate genuinely shared allocations with their
  other active tasks rather than terminating someone else's work.
- This timeout applies to idle retention, not authorized computations still
  making useful progress. Long tests may continue without user messages;
  release their resources at completion if the inactivity deadline has passed.
- Before leaving an idle allocation running, arrange and verify cleanup that
  survives the agent ending its turn or disconnecting. Use a scheduler-enforced
  expiry or a durable job-specific timer, not a promise to check later. A
  reduced Slurm TimeLimit is measured from job start: set total elapsed time
  plus the allowed remaining retention time, not simply TimeLimit=02:00:00.
  Do not assume an unprivileged user can extend a running job again afterward.
- If a durable timer is used, bind it to the recorded managed job and replace
  or disarm it before resuming useful work; new user input can refresh idle
  retention only within the existing allocation and site limits. If reliable
  cleanup cannot be arranged, release the idle allocation before ending the
  turn. Do not rely on an unattended agent to enforce the deadline.

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

The codebase includes the `haplotype_reconstruction/discovery/` modules and several pipeline entry points. Statistical and biological correctness remain paramount, while material runtime, memory, I/O, and scaling constraints are first-class requirements for the real workflow and may justify localized, validated complexity.

The current primary real-data workflow is associated with the cichlid pipeline in `haplotype_reconstruction/workflows/tropheops.py`. Verify this against the current repository before assuming it remains the primary entry point.

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

## Pre-run implementation audit

After implementing a change and before launching a long-running test, expensive
computation, production process, or speculative fan-out, complete a focused
implementation audit. This audit gates launch; deeper empirical validation
gates acceptance.

The pre-run audit must:

1. inspect the final diff and every materially affected direct caller, data
   producer, data consumer, checkpoint boundary, and configuration path;
2. trace the supported execution path that the long run will exercise;
3. check the implementation against its intended biological, statistical, and
   mathematical meaning;
4. examine boundary conditions, indexing and interval conventions, filtering
   and eligibility masks, sample order, missing-data behaviour, output routing,
   resume identities, and completion-marker semantics where relevant;
5. check multiprocessing picklability, process-by-thread limits, shared state,
   worker error propagation, and checkpoint isolation where relevant;
6. run a syntax or import check plus the smallest focused reproducer, unit test,
   or fixture that exercises the changed behaviour;
7. inspect warnings, exceptions, and test output rather than relying only on the
   exit code; and
8. run `git diff --check` and confirm that unrelated working-tree changes were
   not overwritten.

Any known failure that can affect the intended execution path must be fixed
before launch. Do not launch expensive speculative work merely to discover an
implementation error that a focused code audit or short reproducer should have
found.

Keep this launch gate proportionate. It is not a requirement to run broad
simulations, full scientific validation, scaling studies, or preliminary
performance benchmarks. Those should run during the authorized speculative
attempt when safe, and they gate acceptance rather than launch.

Passing the pre-run audit means that no concrete launch-blocking defect was
found; it does not prove that the implementation is bug-free.

## Repository structure

Important entry points currently include:

- `haplotype_reconstruction/workflows/tropheops.py` — primary real cichlid-cross workflow.
- `haplotype_reconstruction/workflows/astcal.py` — another real-cross workflow.
- `haplotype_reconstruction/workflows/simulation.py` — broader or simulation-oriented pipeline driver.
- `run.py simulate` — simulated end-to-end validation against known truth.
- `haplotype_reconstruction/simulation/pedigree.py` — sequence or read simulation.
- `haplotype_reconstruction/recombination/pipeline.py` — downstream recombination-map generation and CLI.

Important infrastructure currently includes:

- `haplotype_reconstruction/core/config.py` — shared model and algorithm configuration.
- `haplotype_reconstruction/core/parallel.py` — process pools, numerical-library limits and dynamic Numba allocation.
- `haplotype_reconstruction/core/checkpoints.py` — compressed checkpoint I/O.
- `haplotype_reconstruction/core/variants.py` — VCF/BCF loading and genotype-likelihood preparation.

The current supported route performs missing-aware block-haplotype discovery,
joint founder completion, component-preserving L1-L4 hierarchical assembly, and
component-local sample painting through typed painting checkpoints. Pedigree inference consumes
those paintings plus raw genotype likelihoods with ragged quadratic scoring,
finite continuous direction, four reciprocal-family cavity-message passes and
a joint short-ancestry correction with a bounded top-16 configuration panel,
followed by a converged final reciprocal solve with the path evidence frozen.
Full-marker raw-GL Mendelian exclusion can resolve M0/M1 counts conditional on
already-supported family directions; it introduces no new edges and preserves
separate bootstrap/LOCO diagnostics. The top-20 candidate-pair panel stays fixed,
and Tier B is primary output.
Family phase performs pedigree-conditioned refinement and genotype-preserving final
phase polishing. Recombination inference consumes final phase for missing-aware, conditional
recombination maps, with separate posterior-mean rates, called crossover
intervals, and observable meiosis exposure. Neither stage feeds back upstream.
The direction/family approximation remains painting-dependent; ambiguous,
missing-parent and highly structured-missingness crosses require validation.

This is a guide, not an authoritative inventory. Inspect the current repository before relying on filenames, stage numbers, or relationships.

## Configuration

Shared tunable model thresholds and feature flags generally belong in `haplotype_reconstruction/core/config.py`.

Before adding a constant:

1. Search for an existing equivalent.
2. Check how related parameters are organised.
3. Confirm it is shared rather than entry-point-specific.
4. Preserve `haplotype_reconstruction/core/config.py` as logic-free if that remains the convention.

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

### Persistent Codex temporary directory

The CSD3 Codex runner uses a persistent, private directory on shared HPC-work
storage. The host-side `/home/ahk39/.codex/.env` contains:

```dotenv
TMPDIR=/rds/user/ahk39/hpc-work/codex-runner-tmp.gKhMfcvc
```

Preserve this setting and directory during cleanup. The random suffix comes
from its original creation with `mktemp -d`; it does not make the directory
automatically expire. This location is separate from the shared project quota
and survives individual compute allocations. It is runner scratch space, not
a location for accepted scientific results or permanent checkpoints.

After a runner restart or host change, verify the effective setting with a
small, normal sandboxed command before launching substantial work:

```bash
hostname
printf 'TMPDIR=%s\n' "${TMPDIR:-unset}"
df -h /rds/user/ahk39/hpc-work/codex-runner-tmp.gKhMfcvc
df -i /rds/user/ahk39/hpc-work/codex-runner-tmp.gKhMfcvc
```

On 16 September 2026, persisting this setting and restarting the remote runner
restored commands after this distinct pre-execution failure:

```text
failed to register synthetic bubblewrap mount target /tmp/.git: No space left on device (os error 28)
```

Do not conflate that error with the namespace-creation error above or infer
that the project filesystem is full from the displayed mount target alone.
Check the actual runner host, its effective temporary directory, and applicable
space/file quotas. A separate login terminal may use a different host or
temporary directory. Inspect only the relevant `TMPDIR` setting, never dump
the complete `.env` file because it may contain credentials.

If the startup setting must change, use an explicitly authorized host-side
change and restart the appropriate remote runner; exporting a variable inside
a command that cannot start is too late. Keep sandbox and approval policies
unchanged. Do not purge live sandbox bookkeeping, kill unrelated runners or
Slurm jobs, or add automatic checkpoint deletion. This persistent workaround
does not guarantee against future quota exhaustion or unrelated sandbox faults.

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
  `core.environment` and `core.parallel` must be imported relative to NumPy and Numba.
- For CPU-bound work that can safely parallelize, target the complete verified
  Slurm CPU affinity. Increase process or thread counts when a larger
  allocation provides useful parallel capacity.
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

`nproc` may honor `OMP_NUM_THREADS` and therefore under-report the available
cpuset. Prefer `os.sched_getaffinity(0)`, the Slurm job record, and the cgroup
cpuset when determining the usable CPU allocation.

Missing Slurm variables do not prove no allocation exists. If absent, inspect `squeue -u "$USER"`, match the hostname, ask when multiple jobs could match, and use `scontrol show job JOB_ID` for authoritative resources.

Do not infer the allocation from physical node size, `/proc/cpuinfo`, or `nproc --all`.

### Resource rules

- Never run substantial computation on a login node.
- Lightweight inspection, syntax checks, and small unit-like tests are acceptable where permitted.
- Complete the mandatory pre-run implementation audit before expensive
  execution. Once it passes, launch the canary and independent provisional
  units concurrently rather than serializing deeper validation.
- Estimate CPU, memory, I/O, and runtime before expensive execution.
- Use only allocated resources. Do not reserve dedicated CPUs for the agent,
  operating system, filesystem, or orchestration.
- Use reviewed `srun` commands where required and reviewed `sbatch` scripts for long or production work.
- Manage task allocations under the standing authorization above; do not
  modify unrelated jobs or bypass site scheduling policy.
- Do not launch the full pipeline unless explicitly requested.
- Avoid tight or repeated polling loops against Slurm; use the bounded
  allocation-management checks described above.

### Performance policy

Performance is a substantive acceptance criterion whenever runtime, peak
memory, I/O volume, or scaling determines whether the intended real-data
workflow is usable. Optimize measured or analytically established bottlenecks,
not hypothetical micro-costs, and validate that performance changes preserve
scientific behaviour.

### Speculative execution and in-run validation

For explicitly authorized computation that decomposes into scientifically
independent units, the mandatory focused implementation audit gates launch,
while deeper scientific, numerical, and performance validation gates
acceptance rather than speculative execution.

After a lightweight preflight, launch concurrently:

- one or more representative canary units selected to expose shared failure
  modes quickly; and
- the remaining independent units as provisional work, sized to use the
  complete verified CPU allocation.

Give the canary enough resources and scheduling priority to finish quickly. Do
not leave CPUs idle merely because its validation is still pending.

Every speculative attempt must have an explicit code, configuration, input,
seed, and checkpoint identity and an isolated output root. Provisional outputs
must not publish global completion markers, overwrite accepted results, or be
treated as accepted until the validation gate passes.

If the canary fails:

1. preserve the minimum logs and evidence needed to diagnose the failure;
2. stop the attempt's remaining workers;
3. invalidate the attempt's provisional outputs;
4. make the required change; and
5. restart under a new attempt identity.

An explicit user request to run speculatively authorizes stopping that
attempt's workers and deleting only outputs created under its validated,
attempt-specific provisional output root. It never authorizes modifying or
deleting pre-existing checkpoints, accepted results, or unrelated outputs.

Speculation is permitted only where scientific and computational dependencies
are already satisfied. Do not speculate across an unresolved global parameter,
shared mutable state, cross-contig dependency, or stage whose inputs may change
when the canary is evaluated.

Performance measurements should normally be collected from the live
speculative run. Do not insert a separate preliminary benchmark when the same
evidence can be obtained from the authorized work.

### Allocation-utilization reassessment

During an authorized long-running computation, periodically reassess whether
the complete verified CPU allocation is being used for useful work.

Perform this reassessment:

- immediately after launching the worker pool;
- when the canary completes or fails;
- at stage, phase, batch, chromosome, or other natural task boundaries;
- when workers exit, the active task count falls, or a straggler tail begins;
- whenever the agent regains control while waiting for a long-running command;
  and
- during a stage lasting more than several minutes when an existing monitoring
  opportunity is available.

Use the verified Slurm CPU affinity or cgroup cpuset as the allocation size; do
not rely on `nproc` when numerical-library thread limits are set.

Compare the allocation against active processes, threads per process, eligible
queued work, and observed useful CPU activity. If a material portion of the
allocation remains idle:

1. determine whether independent eligible work is available;
2. check whether dependencies, memory capacity, memory bandwidth, shared-
   filesystem I/O, or an inherently serial section explains the idle CPUs;
3. if the work is CPU-bound and safe to parallelize, immediately increase the
   worker or thread count, expand remaining tasks into freed cores, or launch
   additional isolated provisional units;
4. keep the aggregate process-count times threads-per-process within the
   verified allocation; and
5. record the underutilization, its cause, and the action taken.

Do not preserve an earlier conservative worker count merely because it was
chosen at launch. Reallocate CPUs dynamically as task availability changes,
especially during straggler tails.

When supported by the validated execution path, use dependency-ready batches
rather than whole chromosomes as the indivisible scheduling unit. As workers
or allocations finish work, let them claim independent batches from unfinished
chromosomes, while retaining node-local dynamic thread reallocation. Choose
work by dependencies, remaining cost and resource fit, not hardcoded chromosome
names or seed-specific exceptions. Preserve scientific batch boundaries,
stable result ordering, single-writer checkpoint ownership and resource limits.
This is a general scheduling preference, not a claim that all current stages
support cross-node batch execution or permission to change a live run to an
unvalidated executor. Planned implementation boundaries are documented in
`docs/performance.md`.

Do not create a separate tight polling loop solely for utilization checks.
Reuse normal command output, worker-pool events, canary decisions, stage
boundaries, and existing monitoring opportunities. If additional concurrency
would not improve useful throughput, continue with the current allocation and
state the concrete limiting factor.

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
- For verified CPU-bound work, default to using the complete Slurm CPU
  affinity.
- When scaling behaviour is unknown, measure it during the live speculative
  attempt wherever outputs can be isolated safely. Use a separate preliminary
  benchmark only when the user explicitly requests one or when a wrong resource
  choice could make the real attempt unsafe rather than merely inefficient.
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
- Keep the aggregate process-count times threads-per-process within the
  verified allocation. Reduce concurrency below that allocation only when
  task count, memory capacity, memory bandwidth, or measured I/O throughput
  prevents useful CPU scaling, not to reserve CPUs for orchestration.

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

- collect elapsed time, CPU use, peak memory, I/O behaviour, worker count, and
  exit status from the authorized live attempt where practical;
- at each natural utilization checkpoint, record allocated CPUs, active
  workers, threads per worker, approximate useful CPU occupancy, idle capacity,
  and any concurrency adjustment made;
- use representative canary units for early validation while other independent
  units run provisionally;
- treat differences between chromosomes, contigs, samples, or regions as
  workload differences rather than controlled worker-count comparisons;
- compare multiple worker counts only when the user explicitly requests tuning
  or when changing a shared default;
- do not create separate benchmark outputs when the same measurements can be
  collected from isolated provisional outputs;
- do not overwrite accepted results or checkpoints;
- report when I/O, memory bandwidth, memory capacity, dependency structure, or
  insufficient task count prevents useful CPU saturation.

Before launching an expensive command, state concisely:

1. the inputs and isolated attempt output root;
2. the verified allocation and aggregate process-by-thread budget;
3. the canary and speculative work decomposition;
4. the acceptance gate and which outputs remain provisional;
5. how this attempt's workers and provisional outputs will be stopped or
   invalidated if the gate fails; and
6. expected runtime, memory, I/O behaviour, and resume semantics.

Once the user has explicitly authorized that run or workflow, do not request
the same approval again and do not interpose a separate benchmark unless a new
material risk or scope change appears.

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

Common artifacts include `*.pkl.b2`, `*.p5.b2`, `.pipeline_checkpoints*`, `results_*`, `logs/`, VCF, BCF, BAM, CRAM, validation CSVs, and run summaries. Check `.gitignore` and `git status` before staging. Never delete checkpoint or result directories without explicit approval.

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

Use a separate small synthetic or bounded test when the user requests one,
when outputs cannot be isolated safely, or when a failure could corrupt
accepted results. Otherwise, for an explicitly authorized decomposable run,
launch a representative canary and independent provisional units concurrently
under an attempt-specific output and checkpoint root.

Do not publish global completion markers or promote provisional outputs until
the applicable validation gate passes.

Checkpointed execution may resume from earlier stages. Before running against
existing checkpoints, confirm that the proposed code is compatible with their
schema, scientific semantics, configuration assumptions, and cached
identities. Before recommending a fresh run, assess checkpoint compatibility
and the cost of recomputation; do not recommend deleting or abandoning
checkpoints merely because compatibility analysis is inconvenient. Never
delete pre-existing or accepted checkpoint or result directories without
explicit approval. Outputs created under an authorized, isolated speculative
attempt may be invalidated or deleted as described above.

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

The supported simulation CLI provides cached known-truth validation through
`python run.py simulate` and `python run.py evaluate`. Historical focused tests
and development fixtures are archived locally, not part of the public package.
Other validation mechanisms include bounded synthetic inputs, held-out founder
comparisons, pair-reconstruction recall, phase/genotype error counts, and
validation CSVs or summaries.

Choose validation according to the change.

### Non-numerical changes

Use focused execution paths, import or syntax checks, small fixtures, schema checks, and regression checks on affected behaviour.

### Numerical or scientific changes

Before accepting numerical or scientific outputs:

1. Identify which results may change and why.
2. Identify relevant metrics. Use an existing baseline or obtain one in
   parallel when practical; do not delay an isolated speculative attempt solely
   to generate a new baseline.
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

Explain non-trivial commands before running them. Require explicit approval
before installing or upgrading software, changing Git history, overwriting
accepted results, modifying the Conda environment, or launching a full pipeline
when the current user request has not already authorized that action. Slurm
allocation management within the standing authorization above does not require
additional approval, including the permitted CPU service-level fallbacks and
release of managed jobs. Actions on unrelated jobs remain outside that scope.

The user's explicit request to launch a run counts as approval for that
specified run. Do not ask for duplicate confirmation. Authorization for an
isolated speculative attempt includes starting and stopping its workers and
discarding only the provisional outputs created under that attempt's validated
output root. Deleting or moving any other files, or recursively scanning large
directories, still requires explicit approval.

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
9. allocated CPUs, utilization checks, workers, threads, sustained idle
   capacity, concurrency adjustments, peak memory, and elapsed time for
   performance work;
10. whether outputs or conclusions depend on unverified biological assumptions;
11. any out-of-scope concern noted but deliberately not pursued.

Do not claim scientific validation solely because code runs or a test passes. Do not propose additional hardening after the requested task is complete unless the user asks for it.
