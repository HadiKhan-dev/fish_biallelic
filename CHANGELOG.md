# Changelog

## 4.6.0 (7 November 2026)

This release makes local selection faster, most at low depth. Configurations, commands, record schemas and table
layouts are those of 4.5.0. Selection objectives, decision rules and fixed points are unchanged, but the optimiser
takes a different path, so selections are not bit-identical to 4.5 below 1x depth. A new output root is required.

**Faster local selection.** On 28 recorded chr1 blocks of the N = 320 simulation, each replayed through its three
selection rounds on one thread, selection wall time per block falls:

| Depth | 4.5.0 | 4.6.0 | Ratio |
|---|---|---|---|
| 5x | 5.5 s | 4.4 s | 0.79 |
| 1x | 14.1 s | 8.8 s | 0.63 |
| 0.5x | 29.7 s | 16.7 s | 0.56 |
| 0.25x | 44.2 s | 23.5 s | 0.53 |
| 0.1x | 67.0 s | 26.7 s | 0.40 |

The changes:
- **One E-step per GEM update.** After a fit's first update (a frequency step), the frequency M-step and the allele
  flips come from one E-step. At each site the founder with the largest expected gain flips; a tied site keeps its
  alleles. The objective splits into a frequency part and an allele part at fixed K, so EM's bound still guarantees
  every rise. 4.5 needed two likelihood passes per update.
- **Batched exact escapes.** At a stationary point, every improving one-bit flip is taken at once, at most one
  founder per site. The batch is kept only if it gains at least as much as the best single flip; otherwise half the
  batch is tried, then the single flip.
- **Batched context snaps.** Snaps are taken together in the same way, with one frequency refit after a run of
  snaps instead of one after each.
- **Accelerated frequency refits.** The fixed-panel refits (trajectory-best and after context snaps) use squared
  extrapolation (SQUAREM), keeping EM's stopping rule.
- **Ranked bank starts.** When a candidate-bank neighbour round has more than 24 new starts, each gets three fit
  iterations, and the best 16 (with ties) are fitted in full.
- **Cheaper exact one-bit gains.** Per-sample likelihood ratios are multiplied in sample order, taking one log per
  founder and site instead of one per sample. The pass costs 0.69-0.84 of 4.5's for K up to 12. The gains agree
  with 4.5's to about 1e-13 nats and do not depend on the thread count.
- **Smaller Python costs.** The set-count code is cached, description costs are read from per-founder tables, and
  thread-team changes are skipped when they would do nothing. These results are bit-identical.

**Effect on selections.** In the same replays (84 block rounds per depth), selections are identical to 4.5 at 5x
and 1x. Below that, rounds that change are more often better than worse under the selection objective:
- 0.5x: 6 better, none worse;
- 0.25x: 20 better, 8 worse (mean +1.1 nats per round);
- 0.1x: 38 better, 29 worse (mean +1.1 nats per round).

Measured and not adopted:
- skipping read-free sites, which pays only below about 0.3x;
- 512-bit vectors, which are bit-identical but 5-20% slower;
- checking exact escapes only at frequency convergence;
- fusing the per-site normalisation sweeps (6-7% of a pass, against the audited normalisation rules).

Fastmath and Intel SVML stay off: results remain the same on every machine and thread count.

**Packaging gate.** `tools/package_clean_environment.sh` runs its install and run cases concurrently, so the gate
takes about 20 minutes instead of 42-53.

**Validation.** Whole-genome N = 320 simulations against 4.4 (whose science 4.5 equals), with the same seed:

| Measure | Depth 1, 4.4 | Depth 1, 4.6 | Depth 0.5, 4.4 | Depth 0.5, 4.6 |
|---|---|---|---|---|
| Founder alleles wrong (per million called) | 5.55 | 5.49 | 79.75 | 78.93 |
| Founder alleles missing (per million) | 19.05 | 19.07 | 235.28 | 237.99 |
| Genotype errors (per million called) | 115.49 | 115.49 | 136.04 | 135.66 |
| Phase switch errors | 31 | 31 | 32 | 32 |
| Pedigree edges correct | 600/600 | 600/600 | 600/600 | 600/600 |
| Parent states correct | 320/320 | 320/320 | 320/320 | 320/320 |
| Core-hours, whole run | 486 | 439 | 760 | 591 |
| Core-hours, selection | 150 | 98 | 386 | 210 |

The test suite (3,708 tests and the thread-team receipt), the whole stage tier (301 passed; two opt-in corpus batches
skipped), the documented workflow example, the strict documentation check and the clean-install packaging gate pass.

## 4.5.0 (7 November 2026)

This release bounds published table cells, makes runs observable and interruptible, gives the simulator a realistic
read model, and shortens the end of a run. Scientific models, decisions, record schemas and released values are
those of 4.4.1. Table layouts change where noted.

**Bounded table cells.**
- Collections whose size grows with the data are published one item per row:
  - painting shadow alternatives: `painting/assembly_search_shadow_alternatives.csv`;
  - symmetric and plateau lists: `painting/assembly_search_items.csv`;
  - cohort-sized pedigree candidate lists: `pedigree/cell_items.csv`;
  - recombination junction members: `recombination/junction_members.csv`;
  - founder-refit excluded samples: `founder_refit/excluded_samples.csv`.
- Parent cells keep counts, and `assembly_search.csv` gains the key `assembly_search_row`.
- Every original collection can be rebuilt exactly; the output guide gives the rules.
- In the N = 320 simulation with 150 bp paired-end reads at depth 1, the largest published cell falls from 161,058
  to 3,938 characters.

**Failed publications leave nothing behind.** Tables, exports and indexes, maps, evaluations, configuration files,
templates, error reports and invocation receipts remove their private temporaries on any exception. Placement remains
atomic.

**Simulation read model.**
- `[simulation.reads]` gains two opt-in settings:
  - `site_depth_cv`: a mean-one site depth factor shared by all samples, optionally constant within
    `site_depth_window_bp`;
  - `count_dispersion`: gamma-Poisson read counts.
- Both work with independent-site and paired-end fragment sampling. Fragment thinning uses local envelopes, so
  candidate fragments scale with local rather than contig-wide maxima.
- `preset = "realistic"` expands to documented values: 150 bp paired-end reads, depth CV 0.3, site CV 0.5 within
  1 kb windows, dispersion 0.1, a 48:52 reference bias and 1% error. Any explicit key wins.
- Existing configurations simulate bit for bit as before.

**Checkpoints survive release-label changes.** The release label moves to a lazily loaded module outside scientific
dependency closures. A release that changes no scientific code keeps every checkpoint identity. Code changes still
change identities, as before.

**Runs are observable.**
- `status` reads per-stage aggregates and cached container indexes; on an N = 320 run it takes seconds instead of
  minutes.
- Run records keep per-stage aggregates; per-task detail stays in the retained `*.tasks.jsonl` log.
- Refinement search-cap diagnostics are logged once per chromosome and stage, not as one warning per component. All
  counts are kept in the level records.
- A progress line reports completed work, running tasks, CPU and memory every 10 minutes and at stage boundaries.

**Runs are interruptible.**
- SIGTERM and SIGINT unwind workers and the coordinator, release shared memory and semaphores, and record the attempt
  as `interrupted` (exit 130).
- Terminated workers get a 5 s grace before a forced kill.
- `report` no longer writes into the checkpoint root: output choices live in `output_selection.json` and its
  completion marker is `OUTPUT/report.done`.

**A shorter end of run.**
- Pedigree resampling runs in work-sized batches.
- Phase starts the largest chromosomes first.
- Recombination and export run on any free worker.
- Per-chromosome export manifests are checked in parallel.
- Export and pedigree writers write independent files concurrently.
- Measured on a whole-node replay of an N = 320 run's tail (pedigree onwards): every published file and
  scientific record is identical, and the tail is 124 s (4%) shorter. Resampling and the longest phase
  chromosome dominate the tail and are unchanged. More concurrent work raised that tail's peak live memory
  from 166 to 198 GB, below the run's earlier-stage peak and within the memory governor's admission.

**Documented evidence, no scientific change.**
- Selection's growth in cost at low depth is diagnosed in `docs/developer/selection_performance.md`: longer exact
  optimisation trajectories, not more candidates.
- Refinement's polish beam widths (64 and 16) hit their bound in thousands of components per run, so they were
  tested. Replays at 2x and 4x width gave identical founders in every component tested, at depths 5, 1 and
  0.5 and in K = 30, except one simulated component where the wider beam was slightly worse. The widths stay,
  and `docs/methods/refinement.md` records the evidence and what a width hit does and does not mean.
- The workflow guide adds the shared-root multi-node recipe.

**Review fixes before release.**
- The pedigree adapter path now gathers resampling batches correctly with more than one worker.
- Per-chromosome tie collectors wait for founder refit as well as painting, so refit tie events are never omitted.
- Progress and failure bookkeeping can no longer fail a run or mask its error.
- Warm-up now covers every kernel overload a run dispatches. Results are unchanged.
  - The refinement exchange-gain sample loops compile on their own and now declare their overloads.
  - The carrier-observation kernel also declares the writable observed mask of pair-coded evidence; a cold run
    had compiled it inside a task, after warm-up.

**Migration.** 4.5.0 changes code and needs a new checkpoint root
([instructions](docs/user/migration.md#migrating-from-44-to-450)). Table consumers must follow the new collection
tables; workflow rules must use `OUTPUT/report.done`.

**Validation.** The 39 cases of the release matrix were rerun with 4.5, and each case's 4.5 collection tables,
converted back to the 4.4 layout, were compared cell by cell with 4.4's: 161,024,422 cells in 1,660 tables and
193,226,256 VCF records, with no scientific difference (only durations and the recorded checkpoint root differ). The
standard N = 320 simulation at depth 2 equals 4.4's in all 24,386,016 table cells and 17,913,704 VCF records. The test
suite (3,693 tests and the thread-team receipt), the whole stage tier, the documented workflow example, the strict
documentation check and the clean-install packaging gate pass. The stage tier had not been run as a whole since 4.0:
its stale tests were brought up to date, external fixture corpora now resolve through one configurable root
(`HR_TEST_CORPORA`), and the 4.0 composition oracle, which tested a frozen 4.0 tree, was removed.

## 4.4.1 (6 November 2026)

This patch release fixes a failure in publishing run-level tables. Scientific models, decisions and every published
value are unchanged from 4.4.0.

**Long table cells.** Run-level tables are assembled from per-chromosome parts written by the report stage. The
parts were read back with Python's `csv` module under its default limit of 131,072 characters per field, a guard
meant for untrusted input. A painting-table cell longer than that stopped the run with `field larger than field
limit (131072)` after every scientific stage had finished. This happened in an N = 320 simulation with 150 bp
paired-end reads at depth 1. The parts are now read without that limit, and the limit is restored afterwards.
Cells under the limit parse and publish exactly as before. Releases 4.3.0 and 4.4.0 have the same limit.

**Migration.** As in every release, checkpoint identities bind the package version, so 4.4.1 needs a new
checkpoint root ([instructions](docs/user/migration.md#migrating-from-440-to-441)). A 4.4.0 run that stopped with
this error must be rerun under 4.4.1.

**Validation.** The test suite passes (3,586 tests, including a new regression test that fails on 4.4.0), as do
the documented workflow example, the strict documentation check and the clean-install packaging gate. The N = 320
run that failed resumed from its checkpoints with this fix alone and published every table and its evaluation.

## 4.4.0 (6 November 2026)

This release reduces retained evidence and execution overhead, and gives idle CPUs to running work. Scientific
models, founder-count decisions, update order, search bounds and released outputs retain 4.3's meaning.

**Memory follows live work.**
- The governor uses a growable request ledger, reclaims idle resident workers under pressure and reports live
  workers after reaping. Waiting work no longer exhausts a fixed request table.
- Shared transport reuses the same backing across aliases, shapes and returned views; mapped inputs retain their
  mappings. This removes duplicate physical copies without merging independent arrays that merely have equal values.
- Evidence schema 2 stores repeated read-count pairs and their exact likelihood triples as per-sample codebooks,
  with checked integer widths and a dense fallback when encoding costs more. Consumers gather their original
  dense numerical windows. Schema-1 format decoding remains supported under its original explicit derivation;
  public 4.4 readers require current derivations. Start a new run/checkpoint root for 4.3-to-4.4 migration
  ([instructions](docs/user/migration.md#migrating-from-43-to-44)). Likelihood precision and masks are unchanged.

**Running work can borrow idle CPUs.**
- Kernel and independent-section boundaries refresh the engine's live CPU authority while preserving a task's
  declared ceiling. Assembly join refits share that authority and retain each kernel's useful-width limit.
- Larger ready assembly joins and components start first. Polish dispatches independent templates dynamically and
  evaluates private bridge trials in byte-bounded waves, restoring logical order before selection.
- Scientific roots wait for their own missing kernel compilations, including assembly's join and refinement tail,
  while unrelated stages proceed. Assembly's stage preload registers its full lazy numerical closure, and the
  supported immutable join signature and Python-called pair helper use the run cache.

**Exact discovery reuse.** Edges with identical intersections of their endpoints' closed neighbourhoods reuse the
same greedy clique extension, with full packed-key equality and the original work receipts. Compatibility tests,
content order, deduplication and whole-size-class admission are unchanged. Repeated dense neighbourhoods avoid
rebuilding the same clique; distinct neighbourhoods retain the original worst-case cost.

**Worker failures.** Error reporting briefly reaps a worker whose pipe has closed, so its known exit status appears
in the failure message; an unavailable status is reported as unknown.

**Checkpoint windows decode whole chunks.** Windowed reads of checkpoint arrays no longer use Blosc2's
partial-range decoding (`SChunk.get_slice`), which in python-blosc2 4.14.1 (c-blosc2 3.3.5) returns wrong bytes
for a range of exactly one internal block that starts inside a block. The store now decompresses whole chunks into
a bounded cache and slices them with NumPy. Releases up to 4.3 used the same call; an audit of every window
actually read by saved 4.3 runs (30 contigs, including the N = 320 simulation) found no wrong value, and full-row
reads, which those releases used for most products, are unaffected. Window reads are faster (median 1.2x for
dense, 2.1x for coded evidence) at a cost of a few hundred MiB of decoded cache.

**Validation.** The 39-case release matrix was rerun with 4.4: every case completes, and all 25 simulations match
4.3's truth scores on every metric (founders, genotypes, phase and pedigree). On the real TroMau and AcTm crosses, the
released outputs are identical under all three noise perturbations and, up to sample order, under both sample
permutations. The standard N = 320 simulation at depth 2, which 4.3 could not complete on a 250 GB node, runs there
in one attempt with a peak of 176 GiB and no memory pacing, and its 24,386,016 table cells and 17,913,704 VCF records
equal 4.3's. At depth 5 every evaluation metric equals 4.3's. Depths 1 and 1.5 complete under a 250 GiB limit.

## 4.3.0 (3 November 2026)

This release tidies the outputs and the code before the analysis freeze. The science is 4.2's: every change leaves
released outputs scientifically identical, apart from one exception path that no validation run reaches (the
read-model fallback below).

**One run-level table per kind.**
- Every published table is one run-level CSV with snake_case columns and a leading `contig` column where it spans
  chromosomes. Contigs follow the configured order; rows keep their order within a contig.
- Shared identifiers use the same names everywhere: `contig`, `sample`, `component`, `founder`, `parent1`,
  `parent2`. JSON outputs keep their keys.
- Painting, quality, recombination and founder-identity tables are concatenated across contigs. Report tasks stay
  parallel per contig; one writer per table publishes it atomically.
- Per-chromosome files remain only for the VCFs and the optional founder tracks (`<contig>.tracks.csv.gz`). The
  recombination figures are the pages of one `recombination/maps.pdf`.
- `haplotypes export RUN DEST` writes the same layout. A TroMau run publishes 134 files instead of 514.
- The [migration guide](docs/user/migration.md) maps every 4.2 path and column to its 4.3 name.

**Recombination coverage is stored once.** The callable spans and the informative spans are equal by construction:
the same consecutive released bins within a stretch, the same ends, components, merge and origin. Recombination
products store them once, as `callable_spans` (schema 2), and the run writes one `coverage.csv`. Map aggregation,
junctions and evaluation read that field.

**Code audit.** Seven reviewers read the whole package for duplication, vestiges, inconsistency, truthfulness and
boundary types. Every accepted change keeps released outputs scientifically identical.
- **Methods text** now matches the code on scoring, evidence filters, posterior release, tie classes, convergence,
  costs and compilation order. No rule changed to match the text.
- **One home each** for state layouts, orientation, nearest-option shadows, pair indexing, painting chunks, the
  phase tie projection, read-only array views, exclusion thresholds, the wide-class guard and evaluation
  vocabularies.
- **Removed:** dead helpers, unused kernel declarations and arguments, an unreleased posterior buffer, unused
  diagnostics and version strings, an empty run-record field, and package functions that only tests called (they
  now live with their tests).
- **Boundary types:** record, array and callback hand-offs declare their types, and the package is type-clean.
- Comments and docstrings describe current behaviour; schema and parameter provenance stay explicit.

**Fixes.**
- `haplotypes run --contigs` supplies the ordered contig list when the configuration has none; with a configured
  list it still selects a subset.
- `haplotypes status` shows selection work in progress.
- A malformed genetic map is reported as an input error, and an unreadable breakpoint file as a settings error.
- Evaluation records the resolved checkpoint root, including a separately configured one.
- When the configured kernel cache directory cannot be used, kernels compile lazily without a disk cache, instead
  of Numba writing cache files beside the installed package's source.
- Read-model calibration keeps the fitted sample-binomial candidate when the later joint fit raises, as the
  [evidence methods](docs/methods/evidence.md) describe; a pooled-fit exception still leaves only the shared
  binomial candidate, and either exception is recorded. None of the 352 saved calibration decisions in the 4.2
  release matrix recorded such an exception.

**Validation.** The 39 cases of the 4.2 release matrix (simulations, and TroMau with its perturbations, both tie
conventions) were rerun with 4.3. Each case's 4.2 outputs, converted to the 4.3 layout, were compared cell by cell
with 4.3's: 161,352,568 cells in 1,660 tables, with no scientific difference. Only durations (704 cells) and the
recorded checkpoint root (64 cells) differ. On matching node types, wall time is unchanged (median 0.98 of 4.2's over
14 cases). The packaging gate installs the wheel and the sdist into fresh environments and reproduces the source
run's scientific records.

Recombination products are schema 2 and code identities change, so 4.3 cannot resume 4.2 checkpoints. Start a new
checkpoint root.

## 4.2.0 (3 November 2026)

This release covers the K = 30 review that 4.0 deferred, and the smaller items left after 4.1. Science changes are
listed first. Every other change leaves released outputs scientifically identical to 4.1.

**Founder-count refits evaluate every hypothesis.**
- The assembly joins' duplicate-seed births, and refinement's deletion and addition rounds, now refit every eligible
  candidate. Before, they refit only the nine best by an unrefitted proxy (`ranked_refits`, cost exemption 0b).
- On the 4.0 release matrix the cap left 1,597 deletion candidates and 111 addition seeds unfitted. Replays on saved
  inputs found two rounds where an omitted candidate fitted better than every admitted one. In both, the count
  verdict was unchanged:
  - a TroMau deletion at K = 10, better by 54.9 nats;
  - an addition in the K = 30 simulation, better by 103.1 nats.
- The decision rule is unchanged: best endpoint F, its tie class and content-smallest member, the symmetric null,
  the bootstrap streams and the shadow families.
- Each count round now fits O(K) candidates, O(N L K³). A pass has at most one more deletion round than it accepts
  deletions, so its worst case is quartic in K. Saved runs had one to five rounds per pass.
- The zero-gain cleanup before refinement's deletion rounds is removed. It fired in none of 54,123 saved component
  refinements, and the ordinary count rule now decides every deletion.
- The tie census loses the shortlist and cleanup kinds.
- Count-rule settings and count-decision records are schema 2.

**Pedigree exclusion evidence is stored as columns.** It used to be one object per screened dyad, about 4 million
at N = 2,500. On a saved N = 2,503 decision seam:
- every rendered table is unchanged;
- the decision step takes 395 s instead of 433 s;
- the seam with its reports takes 405 s instead of 476 s;
- peak memory is 9.7 GB instead of 12.6 GB.

The pedigree product is schema 3.

**Smaller changes.**
- Painting's entry points require the tie-convention reversal explicitly.
- Label continuity groups tie peers once per iteration, so peer lookup is linear in the candidate count.
- Discovery's reader checks each block once on first read.
- The serial re-export route is removed. `haplotypes export` already exported contigs in parallel.
- Duplicated code is removed from phase CAL2, founder identity flags and the painting support tables. Their output
  is byte-identical.
- An unused painting memory setting is removed.
- The example VCF header captures are now `docs/examples/*.header.vcf.txt` and ship with the repository.
- Type declarations are corrected (`JoinSummaryView`, `count_vector_classes`).
- The environment reference lists the thread-limit variables the runtime fixes.

**Reviewed and not changed.**
- **Discovery's bounded searches (exemption 0c):**
  - the replacement starts (48/96) and the archive beam (6/12) bind at every K;
  - they are part of the search's design and need a redesign, not a larger constant;
  - exact gauge cuts certified by component structure are implemented and validated but held back. On their own
    they move three TroMau blocks, two of them to panels the search's own criterion scores lower.
- **Local selection's backstops, bank budgets, context snaps and the LS10/LS17 caps (exemption 0c):** each needs a
  new search or stopping contract.
- **The joins' structural screen (AJ17):** its adaptive route has a cubic analytic bound.
- **Refinement's canonical window fixing (RS05/RS07):** no work stop in any saved run.
- **Painting label continuity and founder-refit tie classes (PR17, FR04):**
  - their worst cases remain above cubic;
  - a refit reorganisation with identical decisions was prototyped, but it does not lower the bound, so it was not
    adopted.

The [cost page](docs/developer/costs.md) and [accretion register](docs/developer/accretions.md) record each finding.

**4.1 validation, completed.** All 39 cases of the 4.1 release matrix finished. Every released output equals 4.0's,
and the ties reports equal 4.0's corrected ones. Wall time, 4.1 over 4.0: median 0.64.

**Validation.** The fast test tier passes (2,914 tests). An independent review of the candidate found no defect.
The release matrix (39 runs: truth simulations, TroMau, references and perturbations) was compared with 4.1:
- **Released outputs identical (32 runs, including TroMau under reversed tie conventions):** only the ties report
  differs (the removed census kinds). In nine of them the assembly search tables and count diagnostics also list fewer
  search events.
- **K = 30 simulation, scored against truth:**
  - Founders improve. One chromosome now releases the true 30 founder rows (4.1 released 32), with called-allele
    errors falling from 1,431 to 603. Founder errors also fall on two other chromosomes.
  - Genotype errors fall from 1,387 to 1,376 and the phased share rises from 0.966 to 0.967.
  - Switch errors move from 30 to 31.
  - Pedigree recovery stays weak in both releases: correct parent states 23 to 24, correct edges 6 to 4.
- **TroMau, the reference run and its five perturbations:**
  - chr3 assembles into one eight-founder component, like 20 of the other 21 chromosomes; 4.1 released three
    components.
  - chr3's recombination junction loci fall from 5 to 3 (none flagged).
  - The primary (Tier B) pedigree is unchanged.
  - Under reversed tie conventions chr3 keeps three components in both releases, so this structure is
    convention-sensitive.
  - Run time on the same node type is unchanged (6,278 s and 6,146 s).
- **Cost of evaluating every count candidate at K = 30:** the K = 30 simulation takes 52.6 minutes against 36.2 for
  4.1, measured on two identical 76-core nodes, with the same peak memory. Both runs reproduce their matrix results
  exactly.

Code identities change, so 4.2 cannot resume 4.1 checkpoints. Start a new checkpoint root.

## 4.1.0 (3 November 2026)

A speed release: scheduling, transport and start-up change; no model, parameter,
threshold, tie rule or output format does. Released outputs are meant to be
scientifically identical to 4.0's. It includes 4.0.1's manuscript
(`paper/`), whose methods are those of this release.

**Start-up.** Kernels are compiled as tasks of the run's own graph, beside the
stages, instead of all of them before the first stage; a stage that meets a kernel
still being compiled waits for that kernel only, and one process at a time writes
each kernel's cache entry. Initialising an observation (VCF/BCF) source runs in the
calling process, without a numerical runtime of its own. The worker server starts
while the graph is planned. On a cold cache the first stage starts within seconds
of the command instead of after the whole warm-up.

**Hand-offs.** A contig's decoded discovery and evidence inputs stay in the worker
that runs its selection and assembly chain, and an expanding task's continuation
stays in that worker instead of travelling through the coordinator. Record decoding
and code-identity hashing are cached per process. A task's first CPU share no
longer borrows from peers that have not taken theirs.

**Tail.** Report tasks run on any free worker; each contig's tie reader follows the
completed stages rather than the exports; recombination runs before export; the
default and mirror phase solves release their samples as one batch; the phased-sample
VCF writer formats a chunk of cells at once, chunks bounded by cells.

**Performance.** On the same nodes (112-core Sapphire Rapids, cold cache, a
116-sample, three-chromosome subset of a real cross), a whole run takes 562-589 s
against 733-777 s for 3.5 (4.0 took 1,265 s where 3.5 took 780 s), and 791-817 s
against 988-1,015 s on 76-core Ice Lake. Whole-run CPU occupancy is about 49% (3.5: 44-46%).
The longest remaining serial stretches are the final refinement level's count pass
on the largest chromosome and each chromosome's evidence reading.

Run records gain each task's `finished_seconds`, and kernel compilations appear as
operational tasks (`kernels/compile/run/<module>:<function>`). Code identities
change with the engine sources, so a 4.1 run cannot resume 4.0 checkpoints: start a
new checkpoint root.

**Validation.** The fast test tier passes. Released outputs on the reference cross
matched between successive candidates. The release matrix (truth simulations,
TroMau, references and perturbations; the hardware-variant rows of 4.0 are dropped)
compares this release with 4.0 and was running when the release was committed. The
workflow receipt (`docs/examples/workflow_receipt.json`) still names the 4.0.0
sources until the workflow integration test is rerun on 4.1.

## 4.0.1 (2 November 2026)

Adds the manuscript describing release 4.0, rebuilt from scratch from the 4.0 methods
documentation and code, in `paper/`: the main text and Supplementary Information
as Markdown sources, the bibliography, figure and table generators, and a build script that writes
the PDF, HTML, DOCX and LaTeX versions. Figure 2 and Supplementary Tables S1 and S2 are drawn
directly from the [cost page](docs/developer/costs.md) and the
[numerical ledger](docs/developer/ties.md). Results that need the outputs of the 4.0 validation
matrix are marked as placeholders in the text.

The scientific code, settings, defaults and outputs are those of 4.0.0. The package's root
initializer carries the version number and is part of every stage's code identity, so 4.0.1 has
new code identities: resuming a run root started under 4.0.0 stops with a checkpoint mismatch
(exit status 2). Finish or resume 4.0.0 runs with a 4.0.0 checkout, and start 4.0.1 runs in a new
output root. For the same reason the workflow receipt (`docs/examples/workflow_receipt.json`) still
names the 4.0.0 sources until the workflow integration test is rerun and the receipt regenerated
with `docs/workflow_receipt.py`.

## 4.0.0 (2 November 2026)

The rebuild separates pure scientific models from runtime orchestration, typed
record publication and output formatting. It adds checked product schemas,
explicit randomness and tie policies, validated storage routes, workflow step
commands and generated documentation. `run CONFIG` still initialises a new run
automatically. Workflow steps require explicit `init` first.

Configurations migrate through `haplotypes migrate-config OLD.toml`: exact
supported translation or an explicit refusal. Unknown keys and removed metadata
are not ignored. A 4.0 run starts with a new checkpoint root; it cannot resume 3.5.
See the [migration guide](docs/user/migration.md) for commands and output changes.

Pairwise `relatedness/pairs.csv` is now optional and off by default (decision 0d).
Changing this output selection does not recompute families. Standalone export no
longer accepts `--vcf`; the stored released-site contract carries source alleles
and contig metadata. Pedigree tables render typed views with preserved text
encodings. Family numbers are canonical; phase records omit obsolete display and
decoder frames.

The [numerical ledger](docs/developer/ties.md#numerical-ledger) describes L1–L15:
unified Haldane fractions and reduction orders, stable duplicate exchange ties,
float64 chain tests, deterministic proof admission and family numbering,
content-based decision sequences, explicit tie settings, deterministic work-stop
witnesses, direct refit gains, map-based context distances, canonical powered
emissions, consistent relatedness move likelihoods and template kernels without
fast arithmetic reassociation, and unphased sample genotypes in ascending allele order.
These are declared numerical changes, not blanket
tolerances for differing scientific output. The owner approved ledger rows L1–L15
and their scoped bounds.

**Validation.** One matrix of 55 runs compares this release with 3.5: truth
simulations, TroMau, references and perturbations. When the release was committed,
30 runs had finished, all with exit status 0, and 29 had been scored. Founders,
paintings, refits, pedigree calls, phase and exports match 3.5. The differences
are declared ledger rows: floating-point drift of at most 1.5e-9 in pedigree
diagnostics (L2, L12), canonical family numbers (L6) and count-step diagnostics
(L7: 3.5 logged repeated rejected death and birth steps that 4.0 skips, so
join-death decision counts are lower by 1 to 28 per run). TroMau gives identical
results at 76 and at 56 cores. The ties report, which omitted discovery's and local
selection's blocks in earlier 4.0 candidates, matches 3.5 apart from those L7
counts. On the same hardware a run takes 1.29-1.36 times 3.5's wall time on TroMau
and 1.5-1.9 times on the simulations.

Retained cost exemptions are context-coded local-selection screening (0), ranked refits
(0b), bounded discovery/selection searches (0c), and optional pairwise IBD
(0d); see the [cost and accretion register](docs/developer/accretions.md).
Four worst-case costs above cubic in K are carried exactly as documented
exceptions (owner decision 0e, the treatment of 0c) and listed for review in 4.1;
the [cost table](docs/developer/costs.md) names them.

The reference platform is Linux x86_64, CPython 3.14.8, with exact runtime pins in
[requirements-tested.txt](requirements-tested.txt). The
[environment reference](docs/reference/environment.md) is generated from those
pins. Packaging requires Python 3.14 or later; the minimum dependency versions are
the tested ones. `llvmlite` and `threadpoolctl` are now declared runtime
dependencies, and `openpyxl` is no longer one. No GPU dependency is introduced.
