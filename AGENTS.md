# AGENTS.md

Rules for every agent working in this repository: Codex, Claude Code (through `CLAUDE.md`) and any other tool. This file
is the single source of shared policy; tool-specific files may add to it, never contradict it. "The owner" owns the
repository and its allocations. An explicit instruction from the owner in the current conversation wins over this file
(point out the conflict). The file states rules; check facts about code, data and cluster against their current state.
A date tag such as [15 Nov] marks an owner ruling of 2026: its words and the incident behind it are in
`AGENTS_HISTORY.md`, read when a rule's reason matters.

## The cardinal rule: one front at a time [16 Nov]

Work on one item until it is resolved: done with its check passed, or blocked with the reason and what unblocks it
recorded. Never open a second front while the first is unresolved, with one exception:
- When the only thing left of an item is a review by another agent, the next item may start.
- When that review arrives, its findings on the earlier item are fixed alongside the next item. On the earlier item
  you only act on the review: its findings and the changes they need.
- If the next item is finished before the earlier item's review is in, wait until that review is done and the earlier
  item is properly closed before starting a third.

So never more than two items are active, and the earlier of two is only ever in review. Waiting on a run is not a reason
to start another item: spend the wait on the item's own next steps. Answering the owner is not a front. Spreading over
several fronts at once (on 16 November 2026: B1, G1i, test re-pins, E6b scoping and docs together) does every one of
them badly and slowly. This rule overrides every other rule about keeping busy (§4.10, §5.4): idle nodes are filled by
the active item's own work, never by opening another front.

## 1. Non-negotiables

1. Never end, shorten, resize or put a cleanup timer on an allocation hosting an agent or its runner, and never signal
   its `screen`, `tmux`, `sshd`, `slurmstepd`, `systemd`, login shell or runner (§6.1).
2. Never stop processes through a user-wide selection: only processes you started, or their verified descendants
   (§7.2).
3. Git history is the owner's. Read-only Git only, unless the owner says otherwise (§4.8).
4. Never delete checkpoints, results or other files without explicit approval, except the provisional outputs of your
   own authorized attempt. Never overwrite accepted results (§5.1, §7.3).
5. Never test on real AulStu (astcal) data (§3.1).
6. Use every node already allocated to the owner, whatever its service level [22 Nov]. Request new allocations on
   SL4-CPU; request SL3-CPU, SL2-CPU or SL3-GPU only when the owner asks at the time, in the size the owner asks for
   [24 Nov]. Running SL4 allocations are kept until they expire (§6.2-§6.4).
7. Whole-genome runs, validation, resume checks and canaries included, use at least one full node (§6.5).
8. Preserve scientific meaning: never silently change a model, objective, threshold, prior, likelihood, filter or
   biological assumption. "Same results" means scientifically identical, not bit-identical (§2.3).
9. Never invent ground truth: cohort labels (G0, F1, F2) do not identify parents, and no sequenced G0 individual is a
   known parent (§2.4).
10. Confirm every run starts and watch it until it ends. Tasks fail fast and never hang (§5.3).
11. Poll `squeue` at most once every 120 s; submit through at most one opportunity every five minutes (§6.6).
12. No installs or Conda changes without approval. Never print `.env` files, credentials or tokens (§3.4, §7.1).
13. Validate a fix where it acts; one end-to-end validation per release candidate (§2.6, §5.5).
14. Never stall on an unanswered question: after 15 minutes proceed with the recommended option. Approvals this file
    requires are never assumed (§4.1).
15. What the owner asked for in a release stays in it; only the owner removes an item (§4.1).
16. The owner's question comes first: answer it at once, on its own, then resume (§4.1).
17. One front at a time (the cardinal rule above: a second only while the first awaits another agent's review), from
    one ordered queue, first item first (§4.10).
18. Parallel by default, tests and lab scripts included [26 Nov]; size each submission to the free cores before
    submitting (§5.4); a serial section needs recorded evidence (§4.6).
19. Test an estimator's criterion against truth before building its machinery. Every option offered to the owner states
    its cost, its falsifying check and its fallback (§2.6, §4.1).

## 2. Science

### 2.1 Mission and priorities

Spend deep reasoning on biological assumptions, statistical models and identifiability, numerical methods and
algorithms, simulation design, validation, scalability and interpretation; not on hardening trusted internals,
adversarial cases, integrity frameworks or abstractions the workflow does not need.

Priority order: (1) biological, statistical and mathematical correctness in supported workflows; (2) scientific
validity and faithful interpretation; (3) reproducibility and faithful handling of inputs and outputs; (4) a working,
testable end-to-end implementation; (5) numerical correctness and stability; (6) performance, memory and I/O on the
real workload; (7) safe, efficient use of CSD3 and Slurm; (8) clarity; (9) minimal complexity; (10) defensive hardening,
only at a real trust boundary or a demonstrated failure.

Performance is first-class when it decides whether the workflow completes at useful scale, but never bought with
correctness, reproducibility or numerical validity. Localized complexity is justified when a simpler version is
demonstrably too slow or large, profiling locates the bottleneck, results are preserved, the complexity stays bounded
and documented, and a reference path remains where practical. Measure and explain trade-offs.

### 2.2 Scientific reasoning

Concentrate on the probabilistic formulation; likelihoods, priors, thresholds and identifiability; genotype likelihoods
at low coverage, haplotype and pedigree ambiguity, phase, recombination and missing data; mathematical versus
behavioural equivalence; known-truth simulations, negative controls, sensitivity, calibration and uncertainty; valid
parallel decomposition; and what metrics and failures mean.

A proposed model change states: the current assumption, the proposed one, why the current behaviour is inadequate,
which outputs may change, how it will be validated independently, and what evidence would falsify it. More elaborate
software never substitutes for missing biological or mathematical justification.

### 2.3 Preserving scientific meaning

- Never silently change algorithms, objectives, convergence criteria, filters, thresholds, priors, likelihoods or
  biological assumptions. Keep numerical-implementation changes distinct from model changes.
- Same results means scientifically identical [20 Oct]: the same called genotypes and missingness, founder alleles and
  counts, selected panels and haplotypes, released ambiguity, pedigree calls and support, phase assignments and phase
  sets, crossovers and intervals; continuous outputs agree well below any meaningful scale. Bit identity is required
  nowhere, refactors and speedups included.
- Floating-point differences from mathematically equivalent computation (another evaluation or reduction order,
  parallel reductions, SIMD, FMA, equivalent kernels) are acceptable, inside a fitted objective or before a threshold
  too [7 Nov]. Never slow down or complicate code to keep bits. Replay on saved inputs, compare released decisions,
  report any threshold flip with its count and margins, and update bit-pinning fixtures (say so).
- A change measured faster with scientifically identical results is kept [16 Nov]: never revert or decline it because
  the gain is small, the bits differ or the bottleneck was elsewhere. Revert only for a reason that would block a new
  change: slower or larger on the realistic workload, a released decision changed beyond this section, a broken
  supported path, or complexity out of proportion to the gain. State the reason and the measurement.
- A small numerical difference that changes a threshold comparison, branch, convergence, selected haplotype, pedigree,
  phase or metric is a behavioural change: report it.
- A better headline metric does not prove correctness. Never change code only to make a test or metric pass, and never
  claim scientific validation because code runs or a test passes.
- State assumptions explicitly. Never present a hypothesis, naming convention, cohort label or inferred relationship as
  ground truth.
- Prefer narrow changes. Before removing apparently unused code, check callers, dynamic imports, checkpoint
  compatibility and diagnostic uses.

### 2.4 Pedigree constraints

- The real cichlid data have no individual-level trio truth unless breeding records prove it. A cohort label (G0, F1,
  F2) does not identify anyone's parents.
- The TroMau cross has four sequenced G0 individuals: AstCal females fAstCal29, fAstCal30 and fAstCal31, and TroMau male
  fTroMau1; every sequenced sample is labelled G0, F1 or F2. No sequenced G0 individual or pair is a known parent; any
  may be a parent of some members and unrelated to others.
- Every sequenced sample of the input is a parent candidate of every other [30 Oct]. The main pipeline reads no sample
  metadata: only the VCF/BCF and the genetic map, sample names being identifiers. Never reintroduce eligibility from
  generation, cohort or sex labels, breeding records or workbooks into `haplotypes run` or any stage; direction comes
  from the data (orientation, family terms, the sibling guard, the release tiers). Lab analyses and reports may use
  labels as descriptions, never as truth.
- Individuals outside a cross's pedigree are left out of the input VCF; in analyses they are never anchors, positive
  controls or inferred relatives.
- Never label pairs or trios true, false, positive, negative or decoy without independent truth. Never calibrate
  thresholds, IBS0, Mendelian error, genotype-likelihood compatibility, founder-label consistency or rankings against
  assumed G0 parentage. Keep species, cohort, known parentage and inferred parentage distinct.
- Without truth, validate through simulations, internal consistency, chromosome-level stability, resampling,
  legitimate negative controls, competing-candidate margins, biologically implausible configurations (labels as
  descriptions), and explicit ambiguous or unresolved outcomes. Preserve missing, single-observed, ambiguous and
  unresolved parents.
- If repository metadata or documentation seems to contradict this, stop and report the evidence first.

### 2.5 Design requirements

One pipeline that works up to K = 30 founders [5 Oct]: no caps sized for small K that break or decide results at
larger K, nothing exponential in K, no stage above O(K^3); prefer better algorithms to budgets. Check K-dependence
(caps, budgets, costs, integer widths) by code analysis and synthetic kernel benchmarks, not full simulations unless
asked. Keep one unified route: remove superseded modes and relict code after the checks in §2.3.

### 2.6 Validation

- **Validate each fix where it acts** [26 Oct]: its reproducing fixture, replays of the affected blocks or components on
  saved checkpoint inputs, competing outputs scored on common inputs, stage-level metrics. Never run the whole pipeline
  to judge one fix: a downstream problem can make a correct fix look worse, and end-to-end runs are slow. A release
  candidate gets exactly one end-to-end validation, once all its fixes are integrated (§5.5).
- Non-numerical changes: focused execution paths, import or syntax checks, small fixtures, schema and regression checks.
- Numerical or scientific changes: say which results may change and why; choose metrics and a baseline (existing, or
  obtained in parallel); hold data, configuration, seeds and resources constant; compare primary and secondary metrics,
  failures, haplotype counts and convergence, missingness, pedigree consistency, phase and switches, runtime, memory,
  warnings and exceptions.
- Recall is not proof. Discrete changes (threshold crossings, rankings, choices, convergence, haplotypes, pedigrees,
  phase, reported outputs) are unacceptable unless intended and validated. Report stochasticity, nondeterminism and
  resource-dependent behaviour.
- Known truth: `haplotypes run CONFIG` with `[input] kind = "simulation"`, then `haplotypes evaluate RUN`; also bounded
  synthetic inputs, held-out founder comparisons, pair-reconstruction recall, phase and genotype error counts.
  Historical focused tests and fixtures are archived locally, outside the public package.
- Pedigree without real-data truth: never report real-data accuracy, precision, recall or false positives against
  truth that does not exist. Simulations for supervised validation; real data for consistency, stability,
  plausibility and failure modes. Report candidate margins, informative-site counts, chromosome support, resampling
  stability, missing-parent states, ambiguity and unresolved outcomes. Avoid circular validation; keep hypotheses
  apart from validated assignments.
- Tests cover supported behaviour and realistic regressions, not adversarial internal states (§4.2).
- **Tests are robust to changes too small to matter** [16 Nov]:
  - Continuous quantities compare within a stated tolerance, well above last-bit drift and well below any meaningful
    scale, with its rationale in a comment. This covers the continuous fields of recorded oracle pins.
  - Discrete outputs (calls, decisions, counts, indices, flags, haplotypes, pedigree, phase) compare exactly. A fixture
    whose decision sits within rounding of a threshold is replaced by one with a clear margin, or the test checks the
    margin.
  - Exact float equality only where it holds by construction (stored and read back, an unchanged copy, an
    integer-valued quantity), said in a comment.
  - Check new or changed numerical tests under a last-bit perturbation (`NPY_DISABLE_CPU_FEATURES`, another thread
    count or node). A test that fails only through last-bit drift is broken: fix the test, never the code.
- **Judge a gate by what its failure means** [22 Nov]: a preset criterion is a tool, not a verdict. A failure that
  evidence shows is noise (a near-tie at a threshold, Monte Carlo or seed variation in a target shared with the
  reference) passes: say so with the evidence and proceed without asking. Bring to the owner only a failure that is a
  real difference in outcome beyond noise (a lost or false edge, a worse metric). Freezing a criterion before the
  results still stands: it stops a criterion being tuned to pass, not judgement about what a failure means.
- **Test an estimator's criterion against truth before building it** [16 Nov]:
  - Before building anything that estimates a setting from the data (a scale, temperature, rate, threshold or weight),
    check on known-truth simulations that the criterion's optimum agrees with the truth's. A lab prototype scoring
    forced values on saved inputs is enough. This check is the first queue item; guards, rounds, diagnostics and
    reviews come after it passes.
  - A criterion computed on part of the problem (held-out prediction on a block, component or pilot) is a proxy when
    the setting acts through later stages: show on several designs that it ranks candidates as the end result does.
    A sharper setting can predict local held-out data better and still make the pipeline worse.
  - If the check fails, report it at once with the evidence and the fallback's cost. The item's fate is the owner's.

## 3. The project

### 3.1 Overview

The package reconstructs founder haplotypes and local ancestry from low-coverage sequencing of experimental crosses.
From a multi-sample VCF/BCF and a genetic map (sample names are identifiers; no metadata [30 Oct]) it estimates pairwise
relatedness and full-sib families from genotype likelihoods, discovers founder haplotypes in marker blocks, assembles
them along chromosomes, paints offspring as diploid founder mosaics, infers the pedigree and corrects phase, and
estimates recombination maps. One pipeline, the `haplotypes` command; stages in graph order: evidence, relatedness,
discovery, local selection, assembly, painting, founder refit, pedigree, family phase, recombination, export.

- The primary real workflow is the TroMau cichlid cross: `haplotypes run CONFIG` with `preset = "tromau"` (preset
  `haplotype_reconstruction/interface/settings/presets/tromau.toml`, example `configs/tropheops.toml`).
- AulStu is never used for testing [17 Oct]. Production runs on it are the owner's decision; there is no preset
  (`configs/astcal.toml` lists its contigs); finished AulStu measurements may be cited as data.

### 3.2 Repository map

- Entry point `haplotypes` (`haplotype_reconstruction/interface/cli.py`, also `python -m haplotype_reconstruction`):
  `run` (initialize if needed, then run or resume the whole graph); the step commands `init`, `evidence --contig`,
  `relatedness`, `reconstruct --contig`, `pedigree`, `phase --contig` and `report` for scattering a run over jobs; and
  `status`, `recombination`, `evaluate`, `export`, `example`, `templates`. Runner `interface/run.py`; public API
  `interface/api.py`.
- Settings: schema `interface/settings/schema.py`, loader `interface/settings/load.py`, presets
  `interface/settings/presets/` (`tromau`, `generic`, `realistic_reads`), examples in `configs/`.
- Known truth: `simulation/`, evaluated by `evaluation/` (§2.6).
- Layers, dependencies pointing down only (the suite's layer test enforces it): `science/<stage>/` (mathematics and
  statistics of each stage, no scheduling, storage or I/O; each has a `parameters.py` with meaning, units and
  rationale); `stages/<stage>/` (orchestration: `tasks.py`, and `records.py`, which owns the keys); `engine/` (task
  graph, scheduler, workers, CPU authority, contig lanes, thread maps, memory governor, run records, Numba
  compilation); `store/` (checkpoint store, codecs, code identities); `contracts/` (typed records between stages);
  `core/` (ties, numerics, digests, errors, files, random streams, kernel declarations); `interface/`; `outputs/`
  (reports, VCF export); `results/`; `evaluation/`; `simulation/`.
- The test suite is not in the repository: the archived 4.3.0 suite (`work/archive/release_4.3.0_tests/`) with each
  candidate tree's `tests/` overlaid, run by `claude_lab/v44lab/harness/run_suite.sh`; a tree's `tests/REMOVED.txt`
  lists archived tests a release retires.
- How the stages fit (`docs/methods/` documents each): relatedness estimates pairwise IBD and full-sib families from
  genotype likelihoods alone, from every contig's evidence before discovery (the relatedness barrier; shards read the
  product their preparatory run recorded); discovery is missing-aware; local selection chooses paths with assembly
  feedback contexts; assembly is hierarchical (L1-L4) with joint founder completion; painting is component-local; the
  pedigree uses paintings and raw genotype likelihoods, Tier B its primary output, every sample a candidate parent, its
  sibling guard withholding edges whose parent looks like one more full sib of the child's family; family phase refines
  phase given the pedigree; recombination estimates maps from the final phase. Family phase and recombination never
  feed back upstream. Direction and family approximations depend on the painting, so ambiguous, missing-parent and
  structured-missingness crosses need validation. Inspect the code before relying on any filename here.

### 3.3 Configuration

- Tunable model constants live in one `*Parameters` record per stage in `science/<stage>/parameters.py`, with meaning,
  units and rationale. User-facing settings live in the TOML schema and map to stage parameters in
  `stages/configure.py`. Before adding a constant, search for an equivalent and follow how related parameters are
  organised. Keep configuration modules logic-free; scatter no unexplained numerical constants through scientific
  modules (established low-level sentinels and capability flags may stay). Dataset paths, output paths and experiment
  selections belong in entry-point configurations.
- `docs/reference/` and `docs/configuration.md` are generated: run `python docs/generate_docs.py` whenever the schema,
  the command line, a preset or a parameter module changes, then `python -P tools/check_docs.py`.
- No backward compatibility before the first public release [14 Nov]: remove superseded settings, outputs, records and
  API names outright, with no deprecation warnings, retired-key tables, migration code, shims or resume support for
  earlier versions; update the lab's configurations, harnesses and tests instead. Reproducibility and resume within one
  version still apply.
- Efficiency beats replay compatibility [16 Nov]: choose the design that is more efficient for the pipeline even when
  it changes a record, identity or checkpoint layout so earlier runs can no longer be resumed or replayed. Never add
  recomputation to the shipped code to keep old runs replayable; replay saved runs in a lab-only prototype tree.

### 3.4 Environment

Use `conda activate /rds/user/ahk39/hpc-work/conda_envs/bio-env` (`conda activate bio-env` only if it resolves there:
verify). Never recreate,
modify, upgrade or install into it without explicit approval; no `sudo`, system packages, unreviewed `pip install` or
`conda install`, or changes to shell start-up files or shared modules. Diagnose dependency problems with focused checks
(`which python`, `python --version`, `sys.executable`, `conda env list`, `PACKAGE.__version__`), never by dumping the
environment. Dependencies: NumPy, Numba, SciPy, pandas, scikit-learn, matplotlib, tqdm, cyvcf2, blosc2, multiprocess;
workflows may call `samtools` or `bcftools`.

## 4. Engineering practice

### 4.1 Task contract, questions and scope

- At the start of non-trivial work, identify the outcome and supported execution path, the files likely to change, the
  scientific quantities that may change, the acceptance criteria and non-goals, the smallest meaningful validation, and
  the performance, memory, I/O or scaling criteria that matter. Ask a focused question when a request is materially
  ambiguous.
- **The owner's question comes first** [15 Nov]. Make no further tool call: the next thing you write is the complete,
  direct answer, starting with the answer and quoting the question. Running tasks keep running. Then resume exactly
  where you stopped (edits, launches, start checks, waiters). Write each answer once; a later reply adds only what is
  new, never a restatement [22 Nov]. A question is not an instruction [22 Nov]: change nothing on the strength of a
  question unless it is plainly a request for that change. If the answer suggests something should change, propose it
  and wait for the owner's confirmation; the 15-minute default does not apply to it.
- **Never stall waiting for an answer** [31 Oct]. Every question states the recommended option and what happens without
  an answer. Keep all independent work moving. After 15 minutes, proceed with the recommended option, record the
  decision and its reasons in the progress file, and tell the owner they can override it. Prefer reversible defaults and
  work a later answer would not waste. Never assume an approval this file requires (Git history, deletion, overwriting
  accepted results, requesting non-SL4 allocations, installs, AulStu, an unrequested full pipeline, removing a
  requested item): leave it pending.
- **A requested release item stays in the release** [15 Nov]. Difficulty changes the plan and the time, never the
  scope: try the next design, measure more, add agents or compute. Only the owner removes, defers or shrinks an item;
  never recommend it as the default option or let a 15-minute default do it. If an item cannot be done soundly or by the
  deadline, say so at once with the evidence, what has been tried, what it would take and when, and keep working while
  the owner decides. The item wins over the date. Every status and completion report lists each requested item with its
  state (done, in progress, blocked and why).
- **Every option carries its cost and its risk** [16 Nov]: its runtime and memory cost on the realistic workload (as a
  share of pipeline time), its implementation time in hours, the cheapest check that could show it unsound and when it
  can report, and the fallback with the fallback's cost. Estimate the cost of an option's sound version when you offer
  it, not after a cheaper version has failed.
- **Stay in scope.** Never widen a task into an audit, redesign, security review, provenance overhaul or cleanup; report
  an unrelated concern briefly and pursue it only if it invalidates the work or the owner agrees. Exception: serial
  bottlenecks in code from release 4.19 on (§4.6). A failure that the item's own runs expose is located within the item
  (its trigger and the stage that first goes wrong); only its fix may become a separate item [22 Nov].
- **Match the process to the change** [5 Oct]: model or decision-rule changes (likelihoods, priors, counting or novelty
  criteria) and algorithm redesigns get diagnosis, a design with prototype runs, and one adversarial review; simple
  fixes get one focused implementation and its focused check. A speedup or operational fix shown to give the same
  results (§2.3) where it acts gets no separate review: that evidence is its check [22 Nov]. Verify what changed rather
  than re-accepting everything.
- **Stop** when the requested behaviour works, the agreed validation passes and the report is ready: no final hardening
  pass, optional abstraction or unrelated refactor.

### 4.2 Threat model and defects

Trusted research code, trusted users and in-process callers; internal dataclasses are not a security boundary. Protect
against data loss, destructive commands, leaked credentials, unsafe handling of genuinely external text in shell
commands, corrupted or partial outputs, and wrong scientific conclusions. Do not defend against deliberate misuse
(`dataclasses.replace`, monkey-patching, mutating frozen objects, forged hashes, hostile pickles); do not add
tamper-evident graphs, parallel identity systems, all-field digests, generalized canonical serialization, schema
registries, immutable wrappers or fail-closed validation frameworks without a concrete current requirement. Hashes are
cache keys, content identities, compatibility markers or provenance aids, not integrity boundaries. Avoid "tampering",
"attack", "trust boundary", "fail closed" and "integrity violation" for ordinary trusted misuse.

Before calling something a defect, establish where practical: a supported or realistically accidental path reaches it;
a concrete call site, input, checkpoint state or user action triggers it; the consequence is observable (wrong output,
wrong branch or selection, crash, data loss, broken documented behaviour, irreproducibility, material waste); and a
reproducer, failing test, trace or code-path evidence supports it.

| Class | Meaning | Action |
|---|---|---|
| Scientific defect | Can alter scientific conclusions or outputs in supported use | Investigate rigorously |
| Operational defect | A real crash, hang, corruption, lost work or unusable workflow | Smallest reliable fix |
| Maintainability issue | Concrete complexity or duplication impeding current work | Only when in scope |
| Speculative hardening | Needs deliberate misuse or a hypothetical need, or has no consequence | Do not implement |

A violated theoretical invariant is not automatically a defect; never force unsupported states in tests to justify new
machinery.

### 4.3 Before editing

Read every applicable instruction file from the repository root to the target (nested `AGENTS.md`,
`AGENTS.override.md`, `CLAUDE.md`, `CLAUDE.local.md`, tool rule files); read the relevant source and its direct callers;
run `git status --short` and inspect uncommitted changes in files you may edit, never overwriting, reverting or
reformatting unrelated changes; for non-trivial work give a short implementation and validation plan; for a purported
defect, meet the defect threshold first (§4.2).

### 4.4 Code style

Match the surrounding code. Keep scientific logic explicit with descriptive names; document units and mathematical
intent. No broad formatting mixed with behavioural changes. Small functions where they clarify, but never fragment hot
numerical kernels; keep Numba-compatible types and control flow. Optimize established bottlenecks without obscuring the
science (no unmeasured micro-optimizations), keeping a reference path for complex optimized kernels where practical. No
dependency without need and approval; no dataset paths in deep scientific modules. Duplication beats a premature
framework: no wrapper, factory, registry, generalized validator or parallel identity scheme for one speculative case.

### 4.5 Concurrency and multiprocessing

- Keep the established start method; everything passed to workers is picklable; worker callbacks live at module scope
  in an importable module, never in `__main__` or inside a function (forkserver workers cannot unpickle them) [21 Oct].
- Numerical libraries and Numba are configured once at package import (`execution/bootstrap.py`) and per run by
  `execution.bootstrap.start_run`. New entry points import through the package and use the `execution` pools and CPU
  budget, never ad hoc pools.
- Keep the BLAS, OpenMP, MKL and Numba oversubscription safeguards; before combining multiprocessing with a threaded
  library or tool, state processes, threads per process and their product, and never nest parallelism without a bound.
- Safely parallel CPU-bound work uses the complete verified CPU affinity (§6.8); widen workers when an allocation grows.
  Before raising concurrency, account for per-worker memory, duplicated arrays, deserialized checkpoints, buffers,
  start-up, serialization and filesystem pressure; choose chunk sizes that avoid excess scheduling, pickling, repeated
  loads and duplication.
- Before changing parallel code, explain the process and thread model, Numba thread scope, determinism and ordering,
  worker exceptions and behaviour under Slurm affinity.

### 4.6 Performance

- **Parallel by default** [15 Nov]: every computation over independent units (samples, sites, windows, bins,
  components, folds, grid values, replicates, chromosomes) runs in parallel from its first version, through a parallel
  kernel, the package's budgeted thread map or independent scheduler tasks (§4.5, §5.4). Plain Python or NumPy over
  large arrays, or a loop over independent fits, inside a multi-threaded task is single-threaded work.
- **Tests and lab work too** [26 Nov]: tests, replays, diagnostics, analyses and lab scripts follow the same rule as
  shipped code. Vectorise over samples, sites and bins (no Python loops over data), fan independent units (contigs,
  arms, shares, folds) out as parallel tasks or a process pool, and size each task to the cores it actually uses and to
  the idle capacity: while whole nodes are free, heavy steps take whole nodes and light per-unit steps share a node
  through a pool. A serial script on a big node, or a narrow task while nodes idle, is a defect of the script.
- **Serial needs evidence**, recorded with numbers in the progress file and a comment at the code: the parallel version
  was not faster on a realistic unit, a stated dependence makes it sequential, or parallelising changes results beyond
  §2.3.
- **Check core use before calling work done**: for each new or changed stage or task, CPU time / (wall time x threads
  granted) on a realistic unit, from its task records; report it, and below about 0.8 find what idles and fix it or
  record the evidence. Results stay the same on any thread count and batch size.
- **Serial bottlenecks** in code from release 4.19 on, or code you are adding, are fixed when found (queued before the
  release candidate, same results shown, core use measured before and after); a large one is reported to the owner at
  once with its cost and date while work continues. Older ones are recorded with their core use for the owner's
  pipeline-wide pass, which you never start yourself. The release report lists both.
- Optimize measured or analytically established bottlenecks, classified first (CPU numerics, compression, filesystem
  I/O, memory bandwidth or capacity, start-up or serialization, an inherently serial step), and validate unchanged
  results (§2.3). Short tests and light checks may stay single-threaded; verified CPU-bound work gets the whole
  affinity.
- Partition only where scientifically valid; never change statistical dependence, random-number semantics, ordering,
  convergence or interpretation without reporting and validating it.
- Avoid several workers decompressing the same large file; prefer indexed queries, bounded regions and streaming; add
  no concurrency when filesystem throughput or memory bandwidth is the limit. Concurrent analyses fit the allocation's
  CPU, memory and I/O together; go below it only for task-count, memory or I/O limits, never to reserve CPUs for
  orchestration. External tools: check that `--threads`/`-@` help (`bcftools` threads mostly speed
  compression); stream through pipes, prefer BCF; `bgzip -@` only when compression is the bottleneck.
- Numba only for stable, repeatedly executed kernels where Python is the profiled bottleneck, never for I/O,
  orchestration or one-off loops; verify that added parallelism parallelizes, cap its threads, keep a reference path,
  compare results.
- Take measurements (elapsed, CPU, peak memory, I/O, workers, exit status) from the authorized live run; a separate
  benchmark only when asked or when a wrong resource choice would make the run unsafe. Compare worker counts only when
  asked or when changing a shared default; differences between chromosomes are workload differences. Report what
  prevents saturation.
  `docs/performance.md` records where each stage stands.

### 4.7 Pre-run audit

Before a long test, expensive computation, production run or fan-out: inspect the final diff and every affected
caller, producer, consumer, checkpoint boundary and configuration path, tracing the path the run will use; check the
code against its scientific meaning; check boundaries, indexing, intervals, eligibility masks, sample order, missing
data, output routing, resume identities and completion markers; check picklability, process x thread limits, shared
state, worker error propagation and checkpoint isolation; run an import or syntax check and the smallest reproducer,
through the real execution path where workers are involved (§5.2), reading warnings and output, not just the exit code;
run `git diff --check` and confirm unrelated changes are intact. Fix every known failure that can affect the run before
launching. Keep it proportionate: passing means no launch-blocking defect was found, not that the code is bug-free.

### 4.8 Git

Read-only Git is allowed for every agent, delegated agents included [10 Nov]: `git status`, `log`, `show` (including
`git show REV:path`), `diff`, `blame`, `grep`, `ls-files`. Everything that changes the working tree, index, refs or
remote is the owner's: never commit, push, pull, merge, rebase, reset, check out, switch, amend, tag, stash, clean or
force, and never discard local changes, without an explicit instruction. After editing, run `git status --short`,
`git diff --stat` and `git diff --check`, and summarize the diff. Never stage generated data, checkpoints, logs or
results. When the owner authorizes a commit or push, commit pending changes to `AGENTS.md`, `AGENTS_HISTORY.md` and
`CLAUDE.md` as their own commit in the same push without asking again [8 Nov], after checking their diff.

### 4.9 Completion report

Report: files changed; behaviour changed; scientific assumptions affected; commands and tests run with exact outcomes;
tests not run and expensive validation still recommended; uncertainties and risks; generated files and jobs; for
performance work, CPUs, utilization, workers, threads, idle capacity, concurrency changes, peak memory and elapsed time;
dependence on unverified biological assumptions; out-of-scope concerns not pursued. Every requested item with its state
(§4.1). Propose no further hardening unless asked.

### 4.10 The work queue

An agent working without delegated agents keeps one ordered queue, `QUEUE.md` beside its progress file [15 Nov]. Each
item states what is to be done and the check that will show it resolved.
- Work on the first item only. An item is resolved when it is done and its check passed, or when it cannot proceed and
  the file records why and what will unblock it; record the outcome, remove it, take the next.
- Insert a new item where its urgency and dependencies put it (before the items that need its result, after those it
  needs), with one line saying why; an owner's instruction goes in the same way, at the front when the owner says it is
  urgent. An owner's question is not an item: answer it at once.
- Waiting: when the first item waits on a running task, mark it `waiting` with its wake signal and the waiter's
  deadline and spend the wait on its own next steps. Only when all that is left of it is another agent's review may
  the next item start (the cardinal rule above); the earlier item then receives only its review fixes.
- A finished run, test or check belongs to the item that launched it, which is not resolved until its result is read
  and recorded.
- Update the file at every start, resolution, insertion, reordering or wait; read it at the start of every turn and
  after every interruption or context summary. Reorder only for a stated reason, written in the file.
- A coordinator keeps the same queue with each item's owner; every delegated agent keeps its own.

## 5. Running computations

### 5.1 Running the pipeline

Verify the current entry point and configuration mechanism. Never edit a production configuration to test, and never
launch the full pipeline unless it was requested. Isolate every attempt: its own output and checkpoint root, and an
explicit code, configuration, input, seed and checkpoint identity; its provisional outputs publish no global completion
marker, overwrite no accepted result, and count only once their validation gate passes. Use a small synthetic or
bounded test instead when asked, when outputs cannot be isolated, or when a failure could corrupt accepted results.
Before resuming from checkpoints, confirm compatibility of schema, scientific meaning, configuration and cached
identities, and weigh it against recomputation; never recommend deleting checkpoints because analysis is
inconvenient. Outputs under an authorized attempt's own root may be invalidated (§5.2); nothing else is deleted
without explicit approval.

### 5.2 Canaries and speculative execution

For authorized work made of independent units, the audit (§4.7) gates the launch and deeper validation gates
acceptance. Launch together one or more representative canaries (given the resources to finish first) and the rest as
provisional work sized to the whole allocation; never leave CPUs idle waiting for canaries. A canary uses the real
execution path (same pool, start method and pickling, at least two worker processes); read its outputs before trusting
the rest. Speculate only where dependencies are satisfied, never across an unresolved global parameter, shared mutable
state, a cross-contig dependency or a stage the canary could change. If a canary fails: keep the minimum logs, stop
the attempt's workers, invalidate its provisional outputs, fix the cause, restart under a new attempt identity. An
explicit request to run speculatively authorizes stopping that attempt's workers and deleting outputs under its root,
never pre-existing checkpoints, accepted results or unrelated files.

Before an expensive launch, state: inputs and attempt root; verified allocation and process x thread budget; canaries
and decomposition; acceptance gate and which outputs stay provisional; how workers and outputs are stopped or
invalidated on failure; expected runtime, memory, I/O and resume behaviour. Once the owner has authorized the run, do
not ask again and insert no separate benchmark unless a new material risk appears.

### 5.3 Every run starts, fails fast and is watched [21 Oct]

- Confirm every start, by you or a delegated agent (local or background process, node-queue task, `srun` step, `sbatch`
  job): within about five minutes check that its processes are alive on the expected node and using CPU, its log shows
  start-up lines and grows, and set-up has finished and the first unit begun. A submit, PID, claim file or `Submitted
  batch job` only shows an attempt. Check again when the first unit should have finished, then at the run's stated
  cadence until it ends, recording each check in the progress file. Do not end a turn straight after a launch: wait,
  sleeping rather than polling, until the start check passes, and leave the next check scheduled. A pending Slurm job
  has not started: follow its expected start and pending reason within §6.6.
- If a run has exited, failed or stopped progressing, read its log at once and fix or report the cause.
- Fail fast, never hang: a pool or background task exits non-zero within seconds when any worker raises, dies or cannot
  start (propagate exceptions, e.g. through `concurrent.futures` or the package's pools); every wait has a timeout or
  liveness check; callbacks are importable (§4.5).
- A task expected to run longer than about 15 minutes writes a heartbeat at least every few minutes and its log keeps
  growing; whoever submits it states the expected duration and the progress signal.
- Judge liveness by progress: at each periodic check compare each task's newest write with its cadence; inspect on its
  node any task silent for about 20 minutes or past its stated heartbeat (process states, CPU, log tail). Near-zero CPU
  and no writes means hung: stop it through its verified process tree (§7.2), keep its log, tell its owner, requeue only
  after the cause is fixed. A stall on a paid allocation is urgent: release that allocation if no other useful work
  remains (SL4 excepted, §6.4).
- An agent ending its turn while a task runs records the task, its expected completion and its progress signal; the
  coordinator resumes on completion, failure or stall and applies the start check to every run a delegated agent
  reports. "No error code" is never evidence that a task is running.

### 5.4 Keep allocations busy with useful work [21 Oct]

- Run worker-node computation through the node queue `nodeq` (`/rds/project/rds-8b3VcZwY7rY/projects/claude_lab/nodeq`,
  see its README) [29 Oct]: `nodeq run` waits and returns the exit code, `nodeq submit` queues; declare `--cores N`
  (`--cores all` for whole-genome runs, §6.5) and `--minutes M`, `--exclusive` for timing checks. Tasks are claimed in
  name order (`--priority`): critical work first, and a short ready queue for nodes about to start. `nodeq status` shows
  idle nodes and why tasks wait; `claude_lab/nodeq/nodeq_remind.py` prints a reminder after 10 idle minutes. Start
  workers with `nodeq/sbatch/nodeq_<partition>.sbatch`; `nodeq/attach.sh JOBID` gives an existing job a worker: the
  hosting allocation, or a node whose superseded `pool_worker.sh` pool has emptied.
- The hosting allocation is a nodeq node too [29-30 Oct]: attach a worker (no `--shared`, no `--cpus` subset) while no
  work started outside nodeq uses its CPUs; it takes whole-node tasks when the hosting job holds the whole node. Its
  tasks fit its free CPUs, memory and walltime, or checkpoint for another worker.
- Fan out independent units, never loop over them in one task [29 Oct]: one task per chromosome, component, sample,
  replicate or parameter value (or small batch), so the slowest unit sets the wall time; lab scripts take the unit as an
  argument and write per-unit outputs plus a merge step. A task uses every core it declares (a process pool over its
  units) or declares only what it uses. Within minutes of submitting, compare busy and declared cores in `nodeq status`:
  long tasks on few busy cores while nodes idle are under-decomposed: cancel and fan out.
- Size every submission to the free cores [16 Nov]: before submitting, read `nodeq status` (idle workers, free cores
  per node, pending tasks) and shape the work to it: split independent units into as many tasks as the free nodes can
  take. On an idle node a task declares the whole node (`--cores all`) and its script uses it (a process pool over its
  units); fewer cores only when no full node is free. Never bundle independent units into one task, or submit a narrow
  task, while cores sit idle. Work that genuinely cannot use more cores is measured once (CPU time / (wall time x
  cores)) and recorded as serial evidence (§4.6). Delegated agents' briefs carry this; the coordinator checks every
  delegated submission against `nodeq status` when it is made, not from the agent's report.
- When a run checks results rather than timing [16 Nov], start its per-chromosome steps on idle nodes from the launch
  (`haplotypes evidence --contig`, `reconstruct --contig` around `calibrate`, `phase --contig` on the same run folder;
  docs/user/workflow.md, "Several nodes, one checkpoint root"); run-wide steps cannot be split. Keep a run on one node
  only when its elapsed time is the measurement, and say so in the progress file.
- Every 30 minutes during a campaign, and when workers start or exit, a canary finishes, a stage or batch ends, a
  straggler tail begins or you regain control: one cheap pass over every allocation (heartbeats, claimed and pending
  tasks, host load, at most one `squeue`), one status line, the same pass serving the stall checks (§5.3). Measure
  against the verified affinity (§6.8).
- Idle capacity (a worker with nothing claimed, a node far below its cores, an allocation with no task): check whether
  dependencies, memory, bandwidth, I/O or a serial step explains it; then queue the next useful authorized work
  (critical path first, then validation that can still change a decision), widen running work, let bounded agent tests
  use a shared host's idle cores, or release non-SL4 allocations (§6.7). Record cause and action, or the limiting factor
  if more concurrency would not help. Reallocate as tasks finish; never keep an early, conservative worker count. Where
  supported, schedule dependency-ready batches rather than whole chromosomes (stable order, single-writer checkpoints).
- Before a turn ends with follow-on work queued, make sure workers whose walltime covers it will exist, or run it on the
  hosting node through a durable waiter. Never fill nodes with busywork (§5.5) or put whole-genome work on a partial
  node (§6.5).

### 5.5 Proportionate validation and superseded work [19 Oct]

- Launch a run only if a plausible outcome could change what happens next. Check the queue first: never queue a test
  whose answer, or a newer version of it, is queued or known. Never rerun checks whose code and inputs are unchanged.
- Validate fixes with focused, bounded tests (§2.6). Broad matrices (every dataset, noise cell, reversal and renaming)
  run once, on the final integrated candidate: the release's only end-to-end validation. Runs of the released baseline
  for the final comparison are not fix validation; say so when proposing them.
- Run the stage tier (`pytest -m stage`, about 300 slow machinery tests on small simulations) only for a release that
  changes kernels or their declarations, evidence storage or windows, threading, batching or worker scheduling,
  checkpoints, resume or publication records, or the workflow and command line [8 Nov]; otherwise run the default suite
  and the workflow receipt, and say which applies and why.
- Cancel superseded work at once: its question is answered, or a newer candidate, tree or configuration replaces what it
  tests (a rejected candidate's matrices, a replaced tree's timings, duplicate checks). A change that forces
  revalidation cancels the old validation in the same step [21 Oct]: remove or hold its pending tasks, stop its running
  tasks through verified process trees (checking for orphans), cancel Slurm requests that served only it, and give the
  freed capacity to the new tasks (running SL4 workers stay until they expire). Keep completed outputs and diagnostic
  logs; record what was cancelled and why. A change that cannot alter any computed result does not supersede
  validation of the same science: say so when keeping such runs.
- In a shared queue, a task's owner asks the coordinator, who acts. Do not busy-wait: end the turn, or sleep without
  re-reading large state.

### 5.6 Durable state, interruptions and the clock

- Keep important work on shared project or home storage; node-local scratch only for reproducible, disposable data.
  Write outputs atomically where practical (temporary file, validate, rename).
- Keep a progress file with each step, its outcome and a `NEXT:` line; check for an existing result before redoing
  work; write results before returning. A cut-off agent is interrupted, not passed or failed. Coordinators keep the
  plan, handoff notes and status log on project storage.
- Every clock time comes from `date` or a tool's timestamp read in the same step that writes it, e.g.
  `echo "- $(date +%H:%M) ..." >> PROGRESS.md` [8 Nov]; never from memory, estimate or a guessed interval. Take
  durations from recorded start and end times; read the clock and the task's recorded start before giving an ETA.
- After an interruption: inspect `git status --short`, `git diff --stat`, `git diff --check`, changed and untracked
  files and known output locations; classify each output as complete, partial, corrupt, absent or uncertain before
  overwriting it; rerun nothing expensive until the last confirmed step is known.

## 6. CSD3 and Slurm

### 6.1 The protected hosting allocation

Never release, cancel, shorten or resize the allocation hosting Codex, Claude Code or their runners, never attach a
cleanup timer to it, and never signal what keeps it alive (its `screen` or `tmux` session, login shell, `sshd`,
`slurmstepd`, `systemd`, the runner), which ends it exactly as `scancel` does. This overrides completion cleanup, cost
optimization and the idle limits; it does not authorize extending it or changing the runner set-up. Before releasing
any allocation, identify and record the hosting job (runner host, Slurm job and cgroup, job list); an unset
`SLURM_JOB_ID` proves nothing, and if ownership is uncertain, touch nothing that might host a runner. Keep the runner
separate from worker allocations where possible; never change the runner or sandbox to acquire resources. A written
handoff never permits releasing it.

### 6.2 What agents may request and manage

On authorized tasks, without asking each time: inspect resources and balances; submit SL4 allocations and run work in
them; resize, release, cancel or replace the allocations they manage. Not covered: unrelated scientific work, an
unrequested full pipeline, anyone else's jobs.
- Request SL4-CPU [17 Oct]. Ask the owner before requesting SL3-CPU, SL2-CPU or SL3-GPU, each time, for that request.
  When SL4 requests wait on the shared budget, leave them pending and say how long the work may slip. Never SL2-GPU or
  any other GPU level. Accounts are normally DURBIN-SL4-CPU, DURBIN-SL3-CPU, DURBIN-SL2-CPU and DURBIN-SL3-GPU; verify
  access.
- Use every node already allocated to the owner, whatever its service level [22 Nov]: the restriction is on requesting,
  never on using. Scripts and wrappers check that they run in a Slurm job with the full node or lease they declare,
  never the account; never refuse, reroute or guard work away from an assigned SL2 or SL3 node.
- Before requesting more: reuse capacity already assigned (coordinating with tasks that share it); optimize useful
  progress per credit and hour (a window too short to finish or checkpoint saves nothing); inspect the user's jobs,
  `hpcstat` (or `sinfo`), the relevant `scontrol` partition, job and QoS records and `mybalance`, reusing recent shared
  observations. Idle nodes alone do not prove a request can start.

### 6.3 Limits

There is no project cap on SL3-CPU and SL2-CPU CPUs [24 Nov]: a request is sized as the owner asks at the time (§6.2).
Never cancel unrelated jobs or the hosting allocation to make room. Every Slurm limit applies (account, association,
QoS, partition, resource, walltime, the shared `GrpTRESRunMins` budget of cores x remaining walltime). Read current
limits; never treat observed free SL4 capacity as a quota. GPU host CPUs must fit the GPU allocation and site limits.

### 6.4 SL4 workers

- Check `hpcstat` at each submission decision [19 Oct] (`bash -lc hpcstat` from a non-login shell). Never queue a
  partition with no free capacity while a suitable one has idle nodes; cancel pending requests on a partition that
  filled while capacity is idle elsewhere.
- Partitions: prefer `icelake-himem` for Ice Lake requests of half a node or more (higher scheduling tier, considered
  before priority); fall back to `icelake` when high-memory nodes are unsuitable or admission favours it, and to
  `sapphire` when it has idle nodes and the Ice Lake partitions do not; record the reason. Small workers may prefer
  `sapphire` when it starts promptly. A full node has 76 CPUs on `icelake` and `icelake-himem`, 112 on `sapphire`
  (verify; history 21 Oct, 29 Sep).
- Request genuine CPU and memory, never inflated memory, counting memory-driven CPUs in the ledger. The default large
  worker is 48 CPUs or a full 76-CPU node for two hours; longer walltimes only for queued work that needs them (e.g. the
  owner's 12-hour validation nodes). Never understate runtime or assume an extension. Use checkpoint and resume for
  longer work.
- Build capacity incrementally: a bounded queue of genuinely useful workers sized to the remaining units, served through
  a shared atomic task queue, admitted by the scheduler. No racing submissions, job churn, admission probes or
  duplicate requests.
- Keep running SL4 allocations until they expire [7 Oct]: never release, cancel or shorten one, by hand, cleanup timer,
  reduced TimeLimit, completion hook or scheduled cancellation; give idle SL4 workers useful authorized work and
  preserve checkpoints across expiry. Never submit SL4 workers no authorized work can use. Pending requests get a
  realistic deadline (wait plus runtime) or durable cancellation; cancel surplus pending workers after counting retained
  allocations and ready and upcoming work.
- Record every job's account, requested and actual cores and memory, submission, start, expiry, task ownership,
  checkpoint state and release action; track useful task time apart from allocated CPU-hours; recover a task claim only
  after its former writer is confirmed terminal; preserve healthy running work when changing future requests.

### 6.5 Whole-genome runs need full nodes [4 Oct]

Never run a whole-genome workload (production, validation, merge-correctness checks, resume tests, canaries, "the last
remaining check", a short resume; delegated agents included) on less than one full node's CPUs. Default: a full 76-CPU
Ice Lake SL4 node (other node types: their verified full-node count), reused across successive tests while its walltime
suffices; the hosting allocation qualifies when it holds a full node. Before launch, record job ID, allocated CPUs,
memory, remaining walltime and worker affinity, and verify the whole node is this run's with no other task (agents on a
hosting node do not count). Pass the full CPU budget through launcher, harness, checker and workers, with process x
thread limits and no hard-coded partial cap; explain any genuine serial, memory or I/O limit. With no full node
available, queue or defer, never fall back to a partial one. Genuinely bounded unit, fixture or single-chromosome checks
may use less; calling a whole-genome run a test does not make it bounded.

### 6.6 Scheduler traffic

Every scheduler call costs controller work (polls, submissions, step launches, cancellations, updates, accounting
queries, retries); another process, node, account or service level is no separate allowance.
- `squeue` at most once every 120 s across all checks and scripts: one call for all relevant jobs, a shared timestamped
  snapshot; poll less when no decision depends on it; never substitute `sacct`, `scontrol` or others to see the same
  state sooner. Focused checks at decision points, batched.
- Aim for about one controller request every two minutes in steady operation; bursts at start-up, recovery or cleanup
  stay bounded, recorded and amortized. Reuse allocations and long-lived workers; distribute units through task queues,
  never a Slurm call per unit. Check the site's rate-limit settings at set-up and reuse them (history 27 Sep); never use
  client backoff as the pacer or probe the remaining allowance. This traffic budget is not an SL4 core cap.
- Grow allocations through one submission opportunity every five minutes, shared by cooperating tasks, only on fresh
  demand: refresh demand from cheap workflow state, count running, pending and in-flight capacity, bound submissions per
  cycle, stop when capacity covers demand. Prefer one demand-sized job array per resource profile. Arrays need an
  array-aware ledger, queue parsing, recovery and cleanup; record array and task identities (elements also have raw job
  IDs that logs may use), count every element that could run against the limits (site array and job limits and QoS
  limits all apply), keep per-element ownership and logs.
- One coordinator runs routine queries and cleanup with shared snapshots and a launch ledger; group queries and
  cancellations; for bulk cancellation prefer `scancel --ctld` with explicit owned selectors and server-side state
  filters such as `--state=PENDING`, excluding the hosting allocation. Ordinary progress comes from heartbeats and
  completion records.
- If calls back off or slow, pause growth while work continues; no rapid retries or replacement queries. Keep scheduler
  and bookkeeping errors apart from scientific failure: pause submissions and record the error while healthy workers
  continue; never broadcast a scientific failure or cancel a campaign over a coordinator exception. Validate changes
  with sparse diagnostics, not a new poller; `sdiag` counts omit some rejected attempts, and its handling time is
  neither CPU time nor harm to other users. Before replacing a coordinator or retrying an ambiguous submission,
  reconcile in-flight submissions; never hold a task-claim lock across a blocking Slurm call; keep capacity reservation
  and ledger updates atomic. Editing documentation does not change a running coordinator.

### 6.7 Reuse, release and idle limits (non-SL4 allocations only)

Not for running SL4 allocations (§6.4) or the hosting allocation (§6.1).
- For a sequence of tests, use one suitably sized and long allocation with headroom and checkpoints. Keep a managed
  allocation through task boundaries when known follow-on work will use its remaining CPUs, memory and walltime [4 Oct];
  record the next use, its readiness and the expiry (each next task fits, or checkpoints, in the remaining time). Reuse
  needs no evidence of contention. Route work explicitly (`srun --jobid=...` or the allocated-node
  session); record each managed job's ID, account, resources, purpose, sharing and expiry.
- Release promptly once authorized and known follow-on work is exhausted, and cancel unneeded pending requests. Retain
  beyond planned reuse only on current evidence that a replacement would be hard to get, recording evidence, reuse,
  cost, job ID and release deadline.
- An idle allocation kept awaiting the owner is released after two hours without owner input in its task (at once if two
  hours have already passed), or at expiry if sooner; agent messages and heartbeats do not reset the clock; an actively
  managed workflow is exempt; coordinate shared allocations with their other tasks. Before leaving one idle, arrange
  durable cleanup (a scheduler-enforced reduced TimeLimit, counted from job start, or a durable job-specific timer,
  disarmed before useful work resumes); a job may not be extendable again; if reliable cleanup cannot be arranged,
  release it before ending the turn.

### 6.8 Where you are and what you have

Determine whether you are in an interactive `sintr` allocation or on an allocated node reached by SSH; never run
substantial computation on a login node. Find the allocation with `hostname`, the `SLURM_*` variables (`JOB_ID`,
`CPUS_ON_NODE`, `CPUS_PER_TASK`, `JOB_CPUS_PER_NODE`, `MEM_PER_NODE`, `MEM_PER_CPU`), `Cpus_allowed_list` in
`/proc/self/status`, `os.sched_getaffinity(0)` and `squeue -u "$USER"`; missing variables prove nothing; read
`scontrol show job JOB_ID` for the authoritative resources and ask when several jobs could match. Size work by the
verified affinity, job record and cgroup cpuset, never `nproc` (it may honour `OMP_NUM_THREADS`), node size,
`/proc/cpuinfo` or `nproc --all`. Use all allocated resources and only those, reserving none for agents, the OS or
orchestration; use reviewed `srun` commands and reviewed `sbatch` scripts for long or production work.

## 7. Shell, processes and data

### 7.1 Commands

Explain non-trivial commands before running them. Get explicit approval before installing or upgrading software or
changing the Conda environment, changing Git history, overwriting accepted results, deleting or moving files outside
your attempt root, recursively scanning large directories, or launching an unrequested full pipeline. Allocation
management within §6.2 needs no further approval; an explicit request to launch a run approves that run; other people's
jobs are never yours. Avoid `rm -rf`, `git clean`, `git reset --hard` and broad wildcard deletion; prefer bounded output
(`tail -n 100`, `grep -n -C 5`, `sed -n 'A,Bp'`). Never expose secrets, credentials, tokens, private keys or unrelated
environment variables, or print a whole `.env` file.

### 7.2 Stopping processes safely [15 Oct]

Stop only processes your task started: record their PIDs, or a dedicated process group, at start. Never choose targets
through a user-wide selection (`ps -u`, `pgrep -u`, `pkill -u`); never use `killall`, `kill -1` or `pkill` (`pkill -f`
also matches its own shell and unrelated command lines); never combine `ps` selection options expecting an
intersection (procps takes the union). `kill 0` or `kill -- -PGID` only for a process group your task created (e.g. with
`setsid`) with a checked ID. Stop a tree from a verified PID through parent links (`pgrep -P` recursively or
`/proc/PID/stat`), excluding your own shell and its ancestors, and check for orphans afterwards. Never signal session
infrastructure (§6.1). Print any computed target list first; if it holds anything but your task's processes, send
nothing. Put these rules in every brief of a delegated agent that may start or stop processes.

### 7.3 Data and generated files

The data are cichlid fish, but the files are large. Inspect the minimum: headers, indexes, sizes, counts, metadata,
bounded samples. Never print thousands of records, recursively scan large trees without approval, copy large datasets
into the repository or invent paths. Summarize large results locally; never expose unnecessary records, sample
identifiers, logs or bulk output to external model context. Never commit datasets, checkpoints, logs or results
(`*.pkl.b2`, `*.p5.b2`, `.pipeline_checkpoints*`, `results_*`, `logs/`, VCF, BCF, BAM, CRAM, validation CSVs, run
summaries); check `.gitignore` and `git status` before staging. Never delete checkpoint or result directories without
explicit approval.

### 7.4 Sandbox on CSD3

- `bwrap: Creating new namespace failed: nesting depth or /proc/sys/user/max_*_namespaces exceeded (ENOSPC)` is a
  namespace failure, not a full disk: the agent already runs inside a user namespace. Run ordinary commands through the
  orchestrator's normal workspace sandbox (in Codex, possibly `csd3-workspace`), never requesting full access or
  escalation to avoid it. Edit with a reviewed unified diff and system `patch` unless a higher-priority instruction
  requires a patch helper; inspect the result and stay inside approved roots. If a required helper fails once with this
  error, use the permitted fallback rather than retrying. Validate as usual, `git diff --check` included. If ordinary
  commands fail the same way, stop and report; never bypass access controls (this file cannot disable an outer sandbox).
- The Codex runner's `TMPDIR` is persistent private storage, set host-side in `/home/ahk39/.codex/.env` as
  `TMPDIR=/rds/project/rds-8b3VcZwY7rY/projects/claude_lab/codex-runner-tmp` [2 Nov]. Preserve the setting and the
  directory during cleanup; it is runner scratch, not a place for results. After a runner restart or host change, check
  `hostname`, `TMPDIR`, and `df -h` and `df -i` of the directory. `failed to register synthetic bubblewrap mount target
  /tmp/.git: No space left on device (os error 28)` is a different error [16 Sep]: check the runner host, its effective
  temporary directory and quotas (a login terminal may differ). Changing the start-up setting needs explicit
  authorization and a runner restart; never purge live sandbox state, kill unrelated runners or jobs, or add automatic
  checkpoint deletion. Inspect only the `TMPDIR` line of `.env`.

## 8. Delegation and orchestration

Use several agents when that improves reasoning, implementation, validation or elapsed time; no cap beyond the
resource rules, but delegation is one level deep.
- Delegated agents do their work themselves and never spawn sub-agents [21 Oct]; launch them with spawning disabled
  where the tool supports it. Their reviews are their own explicit second pass.
- Decompose around concrete biological, mathematical, implementation, code-path or validation questions (roles such as
  scientific-modeler, code-path-explorer, scientific-validator, pragmatic-implementer).
- Every brief carries the task contract (§4.1), the scientific assumptions, paths, acceptance criteria and resource
  limits, and the rules most often broken: safe process stopping (§7.2), no AulStu tests (§3.1), request SL4 but use
  every assigned node (§6.2), read-only Git (§4.8), start checks and fail-fast tasks (§5.3), cancelling superseded
  validation (§5.5), no sub-agents.
- Delegation changes neither scope nor threat model; agents' proposals still meet the defect threshold and the
  validation rules.
- Asking a reviewer never replaces your own work [26 Oct]: work the same problem yourself in parallel (analyse, test
  hypotheses, try to solve it), then compare conclusions. Never wait idle on a reviewer's verdict for a question you
  could investigate.
- The coordinator synthesizes results, reconciles conflicting assumptions, separates evidence from inference, reviews
  every delegated diff and result itself, and stays accountable for the implementation and report. A delegated agent
  that needs compute it cannot submit gets it promptly: its requests are watched, not left waiting.
- Agents return durable outputs (edits, test artifacts, findings, assumptions, uncertainty, the exact next action) and
  keep progress files (§5.6); unreported reasoning is never the only record.
