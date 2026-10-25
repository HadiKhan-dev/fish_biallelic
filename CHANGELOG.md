# Changelog

## 2.8.0 (25 October 2026)

This release rebuilds founder-path refinement in assembly: a robust maximum-path objective of a per-homologue
founder-switching model with map-derived switch costs that reads partially called sites, a symmetric count null,
polynomial per-sweep bounds on every search step, and a final polish of the decided panel with a wider move set.
Outside refinement it adds measured-memory admission, full linker EM steps, a stall watchdog and bounded discovery
and join proposals (**Outside refinement**, below).

**Why.** 2.6/2.7.1 refinement scored panels with a uniform 20-nat penalty per diplotype change and read only sites where
every local row was called; at K = 30 that ignored 18-25% of a component's sites, and the count step missed a founder
the data support (a K = 30 simulation component with 30 true founders: 28 called, 19,744 wrong-or-missing alleles;
2.8 calls 29, 12,589). Its window
programmes had no polynomial bound, and its count null depended on which panel was the incumbent.

**Objective** (`objective.py`, `workspace.py`):
- Each homologue follows its own founder chain; between a sample's consecutive informative sites it resets with
  probability rho = 1 - exp(-h), h = `refinement.reset_rate` (1 per Morgan per homologue) times the genetic-map length
  crossed, and a change costs w = log(1 - rho (1 - 1/K)) - log(rho / K) per homologue. Exact on the K(K+1)/2 unordered
  pairs, O(Q + K) per site. Panel-independent normalisers are added where fits of different K are compared.
- Sites: kept discovery sites with at least one called local row (2.7.1: every local row called). A pair with an
  uncalled row emits the predictive class of the unknown allele(s) under a uniform allele prior; the same uncalled
  row twice is one latent allele.

**Search** (`acceptance.py`, `search.py`, `proposals.py`):
- At most four full re-rankings and four suffix exchanges per junction and pass, then one-at-a-time re-scoring in the
  last ranking's order: every pass is at most cubic in K (2.7.1's rescoring was quartic in the worst case). The
  restarted window programme keeps at most 512 children per unit. Binding bounds are counted and warned.
- Lexicographic (guard) messages: a search's first sweep, and every sweep after one that read switch costs, is
  lexicographic; other sweeps run plain and are rerun lexicographically at the first coarse move above the
  tolerance. Decisions are identical to fully lexicographic sweeps.

**Count step** (`count.py`): the sequence part of every decision uses the larger panel's own canonical death class
(every member evaluated; disagreement or an unresolved member gives the smaller K with ambiguity), so the same two
endpoints get the same call whichever was the incumbent. Accepted additions need no check deletion round; no count
cycle arises.

**Polish** (`stage.py`, `search.py`, `beam.py`): after the count pass, the decided panel and its same-K equal-F
alternatives are refitted once more (one forward-first search) with three move families at every fixed point: a window-beam escape on a
map-aware surrogate, long one-founder moves (64-block beam offers, width 16) with one private correction, and
fine-window/suffix bridges. The polish
never changes K; count decisions are not re-evaluated on the polished panel (recorded). A polish that reaches the
window work bound keeps the count pass's panel.

**Records:** refinement records gain `polish` (status, pre-polish fit and F, gain); fit labels `polish`,
`polish.alt<i>`; new work counters (`content_rounds`, `round_cap`, `stale_evaluated`, `exchange_cap`, `lex_items`,
`lex_abort`, escape and long-repair counters). `count_cycle` is always false.

**Resume:** the discovery, local selection and assembly code identities change, so an output root of an earlier
version raises CheckpointMismatch at discovery; run 2.8.0 into a new output root.

**Outside refinement** (integrated from the 23 October 2.8 components and validated together with the refinement;
see Validation):
- **Linker:** full regularised EM steps (`AssemblyConfig.linker.learning_rate_decay = 1.0`). Uniform initialization,
  the 50-update cap, the fixed-point residual tolerance and the regularisation are unchanged. Finite-iteration
  transition estimates and downstream decisions may change.
- **Measured memory admission:** work is admitted from actual job memory pressure, with no reservations or predicted
  peaks. Below 80% of capacity outer starts are unrestricted; from 80% to 90% each waits for a full observed 0.5 s
  sampling interval; above 90% only the progress exception admits new outer work. Nested continuations proceed so
  admitted work can finish. Scheduling rules only; a delayed or indivisible allocation can still exceed capacity.
- **Tidy-up and watchdog:** lossless int8 sparse flip indices and sizes in completion and founder-allele posteriors
  (K <= 30); unused dense diagnostic helpers removed; abandoned optional memo requests released promptly. A watchdog
  fails a run after 30 minutes of frozen grant/start/release counters with waiters and under one CPU core of activity,
  keeping checkpoints.
- **Discovery proposals:** maximal-clique enumeration is replaced by a content-ordered packed edge-clique cover. Every
  compatible pair is covered before the existing size-class budget, applied per component; larger compatible groups
  can be omitted. The unchanged downstream scoring decides which proposed rows survive.
- **Join proposals:** at most q paths per endpoint, cut ties split in canonical content order; constant-transition
  families use a covering set instead of a Cartesian expansion; the beam-materialization/unjoined fallback is removed.
  These are declared proposal approximations and can change selected or released founders.

### Validation
Component replays: 36 native components, including the six pieces of the equal-resource benchmark, refined from
the same incoming panels as 2.6:

| Component | 2.6 | 2.8 |
|---|---|---|
| K30 m5 (L2; truth 30 founders) | 19,744 / K28 | 12,589 / K29 |
| f3 (L4) | 136,843 | 135,891 |
| stress chr19 [460, 904) | 3,493 | 731 |
| intermediate_K | K7 | K6 (truth 6) |
| K30 L1 / L2 | 2,846 / 11,428 | 2,843 / 11,425 |
| K30 L3 (also the k30_m2 diagnosis ID), m4 | 5,048, 19,187 | 5,053, 19,191 |
| all other 27 component IDs | | same WOM and K |

Validation does not meet the frozen release requirements: f3 per-case and pooled whole-pipeline WOM, fixture trap
rescues and fresh identifiable harms, and per-component count-unverified outcomes (rule 4) fail.

Refinement time on the equal-resource benchmark: 1.93-1.96x 2.6 in two measurements (limit 2x). On whole-pipeline runs
with production warm-up and 2.7.1's memory admission, refinement level walls summed over contigs are 2.14x 2.7.1's
(std 2.47x, stress 2.11x, f3 2.04x, K30 1.92x; backcross 2.10x and TroMau 2.26x measured on a near-identical variant).
Whole runs (to the end of founder refit) take 1.17-1.58x 2.7.1's with this release (std 1,953 s vs 1,236 s, stress
1,751 vs 1,131, f3 1,823 vs 1,173, K30 1,765 vs 1,512) and 1.14-1.34x with the refinement alone: measured-memory
admission does not hold these runs back (no paced admission without memory pressure), but cores stay idle for
longer; the cause is not yet isolated. The 22-contig simulations (refinement alone) take 1.30x and 1.70x 2.7.1's.

Founder-only whole-pipeline matrix (runs stop after founder refit), founder_refit wrong / missing alleles:

| Case | 2.8.0 | 2.6 |
|---|---|---|
| std | 7,120 / 5,208 | 8,385 / 5,207 |
| stress | 1,372 / 2,154 | 3,914 / 2,152 |
| backcross (refinement alone) | 12,409 / 3,927 | 21,518 / 3,926 |
| K30 | 9,068 / 57,203 | 9,109 / 57,142 |
| f3 | 303,129 / 778 | 282,150 / 775 |
| 22 contigs, seed 10000 (refinement alone) | 162 / 159 | 953 / 159 |
| 22 contigs, seed 10001 (refinement alone) | 117 / 17 | 117 / 17 |

- The 22-contig simulations have 320 samples (20 F1, 100 F2, 200 F3). Seed 10000 differs from 2.6 and 2.7.1 on one
  contig (chr3: 1,044 to 253 wrong or missing); K = 6 on all 22 contigs of both seeds, as in 2.6 and 2.7.1.
- The outside-refinement changes alter founder_refit wrong-or-missing alleles by +1 (std), 0 (stress), -179 (K30)
  and +5 (f3); final K is unchanged on every contig.
- f3 is 7.4% worse: on chr2 two founder rows are relabelled across one junction, a labelling that a crossover
  shared by 21 related samples makes the objective prefer (the correct labelling scores 184 nats lower). 2.7.1's
  search reached the correct panel through a different join.
- No new false parent edges (std, stress, K30 and f3 with this release; backcross with the refinement alone). TroMau
  (refinement alone) keeps 2.7.1's founder count on all 22 chromosomes.
- Forced memory pressure (std with a 40 GiB limit, against an unforced peak of about 68 GiB): identical results and no
  stall; the peak reached 44.3 GiB, because admitted work is allowed to finish.
- Fixtures: trap rescues 20/36; identifiable correct-start harms 0/123 (original) and 1/118 (fresh); all
  correct-start harms 99/528 and 81/512 (2.7.1: 233/528 and 220/512); F72 72/72.
- Not rerun with the outside-refinement changes: backcross, TroMau and the 22-contig simulations. Not run: held-out
  simulations, TroMau pseudo-reference consistency.

## 2.7.1 (22 October 2026)

This release restores founder-path refinement in assembly, which 2.7.0 removed.

**Why.** With refinement off, the 2.7.0 founder-only validation on simulations with known truth failed its gate. Pooled
native-refit founder wrong-or-missing rose from 394,278 to 671,099 (+70%; gate at most +1%):

| Case | 2.6 | 2.7.0 | Change |
|---|---|---|---|
| std | 13,592 | 30,670 | +126% |
| stress | 6,066 | 81,868 | +1,250% |
| f3 | 282,925 | 437,777 | +55% |
| backcross | 25,444 | 53,668 | +111% |
| K30 | 66,251 | 67,116 | +1.3% |

- **Cause:** surplus founder rows left by the joins survived. Several chromosomes ended with K = 7-9 where the truth
  has 6, which split true founders across rows; refinement's count step removes them.
- **f3:** gained one new false parent edge.
- **TroMau:** native pseudo-reference mismatches fell (615,274 -> 470,063). That is consistency with a dependent
  pseudo-reference, not accuracy: extra founder rows match its eight G0 haplotypes more easily.

**What 2.7.1 keeps from 2.7.0:**
- Founder completion's exact O(K^3) reorganisation, with identical released results. W3's identity runs on std, K30
  and TroMau were identical with refinement on.
- The linker EM update cap of 50. Fits that stopped at the old cap of 20 can change.
- Deterministic seed-family promotion order in joins, and removal of the unused `shadow_bounds._join` and
  `join_envelope`.

**What 2.7.1 restores from 2.6:**
- The 2.6 refinement model, with its count step, records and reports: `founder_counts.tsv` comes from refinement's
  count pass again.
- The 2.6 documentation of these.

A new refinement is being designed for 2.8.

**Resume:** assembly code identity changes, so 2.7.0 output roots are recomputed from assembly onward, not resumed.

**Operational fix: a memory-governor deadlock in founder refinement.**
- **Symptom:** 22-contig runs with refinement on could stall indefinitely. Reservations sat near the limit while real
  memory was free, and waiters asked for 0.01-0.03 GiB with almost no CPU in use.
- **Cause, present since 2.6:** a concurrent task whose memory was granted while it still waited for a CPU slot was
  marked as running work. That disabled the governor's rule of admitting the head of the queue when every holder is
  waiting.
- **Why 2.7 exposed it:** larger refinement memory estimates (the guard traceback term) and long-held worker
  reservations made the state reachable.
- **Fix, scheduling only:**
  - granted-but-unstarted tasks stay queued until they start (`execution/budget.py`, `execution/memory.py`);
  - guard scratch is reserved only around guarded acceptance (`assembly/founder_refinement/{search,acceptance}.py`).
  Both change when work is admitted, never what is computed. The guard-scratch change was checked identical on
  backcross, N640 and TroMau, and a bounded reproducer hangs before the fix and completes after it.


**Known limitation, unchanged:** family phase, the recombination maps derived from it and phased-sample VCFs are known
to be broken and are not validated.

### Validation

Founder-only matrix (runs stop after founder refit; family phase excluded), against 2.6. Each cell is founder_refit
wrong / missing alleles.

| Case | 2.7.1 | 2.6 |
|---|---|---|
| std | 8,385 / 5,207 | identical |
| stress | 3,914 / 2,152 | identical |
| backcross | 21,518 / 3,926 | identical |
| s10000 (22 contigs) | 953 / 159 | identical |
| s10001 (22 contigs) | 117 / 17 | identical |
| f3 | 282,249 / 770 | 282,150 / 775 |
| K30 | 9,048 / 57,339 | 9,109 / 57,142 |

- Every s10000 and s10001 chromosome has K = 6 in one piece.
- Pooled wrong-or-missing over the seven simulations: +0.06%. No new false parent edges; f3 keeps 2.6's eight.
- The f3 and K30 differences come from the linker cap of 50.
- TroMau native pseudo-reference mismatches: 596,014 against 615,274 (−3.1%; consistency only, not accuracy).
- The governor-deadlock fix changes scheduling only. std and TroMau rerun with it are identical, per chromosome, to the
  matrix runs above.

## 2.7.0 (22 October 2026; superseded by 2.7.1, do not use)

This release retains founder completion, removes founder-path refinement from assembly, and raises the linker EM
update cap to 50. These owner-directed changes can alter scientific results. Final founder validation is pending.

**Known limitation:** family phase, the recombination maps derived from it and phased-sample VCFs are known to be
broken and are not validated in 2.7. Those stages still run by default, unchanged from 2.6. Validation stops after
founder refit and assesses per-chromosome founder concordance, parent edges and TroMau pseudo-reference consistency.

### Assembly and count reporting

- Joined components and their own release alternatives pass directly between assembly levels and into the final
  assembly product. Founder-path refinement is disabled; its source is retained outside the pipeline and kernel
  warm-up for the 2.8 rebuild. Downstream founder allele refit remains active and is a distinct stage.
- Count reports use the latest actual join, distinguish incomplete verification from demonstrated ambiguity, and
  identify local panels that have never joined. The latest join is included only once in cycle totals. There is no
  new final count-addition pass or heuristic dead band for near-zero count margins.
- The final founder evaluation stage is `assembly`. Assembly schema 3 and its stored configuration reject refined
  checkpoints before completed-stage shortcuts, including the superseded schema-3 completion-drop candidate.
- Ambiguity records from every accepted seed reaching the selected canonical panel are retained. Accepted-seed
  histories are promoted in founder-path order, so thread completion order does not order their witness families.
  Two unused shadow-count helpers were removed; the separate full-site proof remains.
- Discovery I6 remains a documented limitation, not a new inference rule. No unobserved founder allele is selected
  by a new convention in this release.

### Exact computational changes and limits

- Packed symmetric emissions, reused splice and count-cover calculations, compact beam families and retained-work
  memory accounting reduce selected costs without changing their numerical objectives. Refinement-only speed and
  window-policy work is inactive; no refinement window limit governs the shipped assembly path.
- Completion represents growing support with sparse flips instead of a configuration-by-founder matrix. At fixed
  sample count and block length it is O(K^3); exact cubes with at most ten uncertain alleles are bounded by 1,024
  configurations. The small-cube cavity cancellation fallback is O(N 2^U K^2). Independent focused benchmarks fit
  EM dosage exponents 4.28 to 2.89, configuration updates 4.03 to 2.68 and cross-fit blocks 3.33 to 1.99. No measured
  memory reduction at K<=30, or universal whole-pipeline speedup, is claimed. Full completion identity gates remain
  pending until the std, K30 and TroMau comparisons finish.
- Complete two-row exclusion domains use adaptive segment evaluation or direct forwards. At production bin counts
  and K<=30 the direct route is used, and this term grows approximately as K^4 until K^2 is roughly 3m or larger.
  The cubic segment route does not establish a cubic bound for every production workload or the whole join search.
- Any binding beam allowance can change outputs. A constant family beyond the allowance releases the unsearched
  anchor panel without masking, flagged in metadata; a nonconstant family retains the incumbent or leaves the inputs
  unjoined. Earlier validation observed no binding. This remains a quality risk for the 2.8 work; it is not an exact
  alternative to searching the domain.
- A packaged kernel manifest supplements learned store signatures, compiling release overloads once in the warm-up
  pool. Disabled or removed kernels are skipped and the 2.6 cache-signature check remains. The 508.9 s cold N80 result
  belongs to c0 plus the W2 manifest change, not this final tree; no final-tree cold timing is claimed.

### Linker convergence

- The residual is checked at the returned transition matrix, adding an E-step per fit. W4's bounded fixtures used
  2% to 29% more updates under that stopping rule. This correction can change outputs.
- The owner raised the update cap from 20 to 50. Tolerance 0.001, initialization and damping remain unchanged; at most
  51 E/M evaluations include the returned-iterate check. A fit at the cap may still be unconverged, and its residual
  is reported honestly. On one saved K30 gap, the longer fit converged at update 34; this is not a whole-run guarantee.
- Cap hits were common in the earlier cap-20 c2 keep matrix: 11,187/24,170 TroMau gap fits (46%), 417/561 K30 fits
  (74%) and 6,470/24,875 std fits (26%) stopped above tolerance. These are not cap-50 frequencies. The final matrix
  supplies the cap-50 census.

### Validation and deferred work

- Final founder-only runs on this refinement-off tree FAILED the pooled gate (+70% founder wrong-or-missing; see
  2.7.1). Gates are pooled native-refit founder wrong-or-missing
  at most +1%; no simulation's native wrong calls increasing both >5% and >200 cells; no new false parent edges;
  and TroMau native-refit pseudo-reference mismatches at most +1%. TroMau consistency is not accuracy against known
  parents or true founder haplotypes. Offspring genotype and phase-switch gates are excluded by the owner.
- Per-chromosome founder concordance with the true generating haplotypes, s10000 and s10001 (22 chromosomes each, 6
  true founders): with refinement on (the earlier candidate), every chromosome is one piece with all 6 founders found
  and matched, and 953/159 (s10000) and 117/17 (s10001) wrong/missing alleles of 53,741,112, identical to 2.6.
  The refinement-off tree did not reach these: see 2.7.1.
- The earlier f3_fam matrix inherited eight false parent edges unchanged from 2.6. They were not new errors; the
  final matrix must still report their identities and any new edges explicitly.
- Export manifests contain wall-clock release diagnostics and are not byte reproducible; those diagnostics do not
  alter VCF content. This does not validate the known-broken phased-sample outputs.
- Release continuity, shared-junction effects in joins, missing K30 founders without additions, refinement redesign
  and linker acceleration are deferred to 2.8. No W4 continuity fix or refinement-memory overlay is included here.

## 2.6.0 (21 October 2026)

Release 2.6 speeds discovery and local selection, and separates the copying intensity from assembly's gap horizon.
Both parameter defaults stay at 3, preserving scientific results; the local-selection conventions remain unchanged.

### Local selection: conventions checked

- The bank's 512 + 64K start cap binds in six block-rounds on TroMau and K30. There are 73 same-K
  distinct-winner ties, six of which advance the inner search.
- Reversal checks change no released local call, K, missingness, support or decision in 223 flagged
  block-round replays or through all three rounds on K30 m1 and TroMau chr7. One uncalled latent cell
  and posterior drift up to 7e-6 remain on K30. No remedy is needed on this evidence.

### Copying and assembly parameters separated

- The shared `n_generations` is replaced by two independent internal parameters:
  `LocalSelectionConfig.model.redraw_rate_per_morgan = 3.0` is the founder-copy redraw intensity per Morgan;
  `AssemblyConfig.gap_horizon_scale = 3` limits linker and beam proposal gaps. Defaults retain the same
  arithmetic and scientific results. Neither parameter is a public TOML setting.
- In a bounded chr19 sweep, copying intensities 1 and 6 worsen TroMau G0 pseudo-truth consistency.
  Other settings bring no principled gain across std, F3 and TroMau. Both defaults stay at 3;
  no generation labels or painting-switch estimates set them.

### Discovery speed

- Birth proposals retain canonical keys and descriptors until fitting needs panel arrays. Fractional-founder
  scalar certification runs before term-column views are built. Search budgets, objectives and fits are unchanged.
- Matched single-contig discovery wall times on 76-core Ice Lake nodes fall from 33.3 to 30.1 s at N=80
  (-9.6%), 54.4 to 47.0 s at N=320 (-13.6%), and 32.7 to 30.6 s on TroMau (-6.4%). All 2,696 blocks
  and fit counts agree, with zero measured continuous drift. Each result is one matched pair, not a
  whole-pipeline timing estimate.

### Local-selection speed

- Copying kernels use adaptive sample lanes and cap their threads at the useful lane count. Normalization
  reuses reciprocals and combines triangular normalization with marginal accumulation. The model, search
  budgets, thresholds and stopping rules are unchanged.
- Matched local-stage wall times on a 76-core Ice Lake node fall from 373.3 to 237.1 s on K30 m1 (-36.5%),
  200.8 to 179.8 s on TroMau chr7 (-10.5%), and 120.2 to 114.3 s on std chr1 (-4.9%). N320 is within
  timing noise. These are single matched pairs, not whole-pipeline speedups; K30 uses about 7% more CPU.
- Released scientific fields agree in four matched local comparisons and their final assembly/painting checks.
  Full std passes local selection, assembly, painting, pedigree, founder refit, family phase, recombination,
  truth evaluation and all 4,727,400 records in 16 exported VCFs. Maximum local log-likelihood drift is
  2e-10; final products, evaluation and exports have zero measured drift. Kernel and parameter-split checks pass.

### Findings, no code change

- Targeted 2x/4x discovery-budget replays show mixed accuracy effects and no demonstrated downstream
  accuracy gain. Search budgets are unchanged.
- A conditional residual-seed augmentation was prototyped and rejected: final K30 genotype wrong-or-missing
  rose 1.7%, exceeding the allowed 1%. No rare-founder code change is included.
- The global tie-reversal hook changes one TroMau L2 context-assembly `release_alternatives` record.
  The isolated local winner reversal does not cause it, and released local calls remain unchanged.
  This assembly convention remains a follow-up for the assembly stage.

### Fixed

- A cold-cache crash present since 2.0: parallel kernel warm-up could leave Numba's cache index pointing
  at another overload's compiled code, causing the caller to fail to type. This occurred in a K30 simulation.
  Cached kernels are used only when their signature matches the requested one; otherwise they are recompiled.

### Validation

- Whole-pipeline runs on 76-core Ice Lake nodes agree with 2.5 in every released discovery field, every
  pedigree call, and the per-contig site counts, K, painting switches and crossovers:
  - TroMau: 9,979 blocks, 116 samples, 22 contigs;
  - the std, stress, K20, K30, f3 and reallike simulations;
  - the N-series at N = 80, 160, 320 and 640.
- Whole-pipeline wall times, 2.5 against 2.6:
  - TroMau 67.7 against 64.7 min; std 24.6 against 22.6 min;
  - N-series 10.6, 14.6, 22.6 and 45.6 against 8.2, 13.2, 20.8 and 46.1 min;
  - matched pairs run side by side: K30 46.1 against 39.0 min (-15%), K20 7.3 against 6.8 min, f3 22.7
    against 22.0 min, stress 21.7 against 21.6 min and reallike 17.4 against 18.1 min.
- A reproducer with a corrupted kernel-cache index fails on 2.5 and passes on 2.6, both in the main process
  and in pool workers. The focused tests pass.

## 2.5.0 (21 October 2026)

Release 2.5 changes the discovery stage. Discovery is practical at large sample sizes, and population-half
proposal starts are independent of any hardening convention. Released results on TroMau and the simulations are
those of 2.3/2.4 except in rare population-half blocks.

### Discovery speed at large N
- 2.3 and 2.4 had a residual-group tail. At N=320 a few blocks took tens of minutes each: chr1 of an N=320
  simulation took 2,476.9 s against 79.1 s in 2.2, and the tail made whole runs impractical. Residual alternatives
  are now counted and deduplicated as exact packed original-record memberships before Python groups are built.
  Component admission, complete size classes, omissions and every released decision are unchanged. The adverse
  N=320 block falls from 1,625.9 s and 54.4 GiB to 9.8 s and 2.1 GiB.
- Small residual pools and clique counts run inline. Larger independent pools share the live CPU budget, so
  freed cores pass to unfinished work.
- Fixed-K fitting uses guarded fractional-founder reductions. Wherever a decision could depend on them, the
  accurate-sum path decides.
- Discovery hands borrowed CPU threads to waiting block workers at its existing claim boundaries. Other stages
  keep their scheduling policy.
- Discovery wall time on the five N-series chromosomes (76-core Ice Lake nodes):
  - N=160: 7,973 s in 2.4, 235 s in 2.5 (34x faster);
  - N=320: 10,266 s in 2.4, 369 s in 2.5 (28x faster);
  - std_n80 (eight chromosomes): 367 s in 2.3, 331 s in 2.5.

### Population-half proposals
- Both coherent hardenings of population-half ties are evaluated as one proposal family, and duplicate starts
  are removed before fitting and admission. Budgets and co-winning alternatives are unchanged. Blocks without
  such ties are unchanged. On TroMau one block of 9,979 changes its panel, and the change does not reach the
  assembled founders, paintings or pedigree. Truth effects on low-coverage tests are small and mixed, and no
  accuracy gain is claimed.

### Diagnostics
- Expansion and score backstops are reported separately from ranked truncation and omitted tied classes.
- Search records persist archive-maximum K, block and phase wall and CPU times, and selected versus
  other-panel mean-field failures.
- Completed discovery blocks and their diagnostics are checkpointed every 64 blocks, or on the first completion
  at least 30 seconds after the last flush, so a stopped run keeps nearly all finished blocks.

### Removed modes
- The optional reversed-convention second search and its escalation are removed. The default search, the
  dense-geometry reference and acceptance-only reversal checks remain. Historical mode metadata still decodes.

### Findings
- Landmark (A256) and dense residual geometry agree in all 60 tested comparisons, including 20 landmark-active
  N=320 blocks.
- The TroMau chr1 and chr23 pseudo-truth regressions from 2.2 to 2.3 come mainly from long-range identity
  propagation in chromosome assembly. Block-level discovery is about equal on chr1 and better on chr23.
  Assembly is outside this release.
- Simulated K30 is under-K in 200 of 200 blocks, driven by rare founders: an information limit at N=80. K6 misses
  are near-twin founders. No tested wrong K lies within the score quantum. A broader complete-family eligibility
  policy worsened all four tested cases, so the selection objective is unchanged.
- A normalized shared-sequence panel prior was implemented exactly, including unobserved founder cells, and
  validated end to end. Gains on K30 were large: genotype wrong-or-missing -28% and founder called errors -2,911. Across seven
  held-out simulations, genotype wrong-or-missing fell 10.6%.
  But it failed the predeclared gates:
  - final founder accuracy was worse at N=80, with wrong-or-missing alleles +24%;
  - surplus-K blocks increased at N=80, at N=320 and in the backcross;
  - on K30 the called-genotype error rate rose 12% and phase switch errors rose from 192 to 272;
  - TroMau mismatches against the G0 pseudo-truth rose 5.5%.
  It is not included, and it remains the leading research item for high-K designs.
- 2.2 against 2.3 on seven paired simulations:
  - common-called genotype errors are identical in five;
  - K30 and backcross trade small genotype changes against founder recovery and phase continuity;
  - phase sets are more and shorter;
  - f3 gains four wrong parent edges.

  2.3 is therefore not more accurate overall. On TroMau, mismatches against the G0 v1.4 pseudo-truth fall
  2.8%, with chr1 and chr23 worse. These later-stage regressions are follow-ups for assembly, pedigree and
  family phase.

### Operational fix
- A cache-order-dependent Numba typing crash in family phase is fixed by passing consistent scalar types to the
  cached frame, block, cavity and polish kernels. Scientific outputs are unchanged.

### Validation
- **Scientific identity with 2.3 on TroMau:** all 22 chromosomes have identical sites, K, painting switches and
  crossovers, and no pedigree parent call changes.
- **Simulations:** std, stress and K20 show no released discovery difference from 2.3. Measured accuracy equals
  2.3 on all seven paired simulations (std, stress, K20, K30, f3, real-like and backcross). N=160 and N=320 discovery
  match 2.4 block for block.
- **Speed work is exact:** successive candidates reproduce every discovery field except timing, plus pedigree,
  K, switches and crossovers. On a three-chromosome check the exported founder and sample VCFs are byte-identical.
- **Whole-pipeline scaling (N-series, five chromosomes):** 10.6, 14.6, 22.6 and 45.6 minutes at N = 80, 160, 320
  and 640. The N=80 run includes about 2 minutes of first-use kernel compilation. The scaling exponent is about 0.8.
- **TroMau whole pipeline:** 67.7 minutes, against 72.4 minutes for 2.4.
- **Final accuracy across N (N-series):**
  - founder haplotypes are reconstructed without error at N >= 160; N=80 has 1,021 wrong-or-missing alleles per
    million;
  - genotype errors are 165-194 per million called genotypes;
  - phase switches are about 1 per million heterozygous sites;
  - there are no false parent edges.
- **Sample order:** TroMau with a permuted VCF sample order gives identical outputs.

## 2.4.0 (20 October 2026)

Release 2.4 changes the evidence stage only. Evidence products on all 22 TroMau chromosomes are identical to 2.3,
so released TroMau results are unchanged.

### Site filter: what it removes, and the heterozygote-balance rule
- The site filter keeps 33.7% of TroMau's biallelic SNPs (1,993,684 of 5,918,884). A diagnosis on chr1, chr3 and
  chr8 shows the removed sites are not clean sites lost to over-filtering:
  - At heterozygous genotypes they carry marked reference bias: the heterozygote ALT fraction is 0.38-0.43,
    against 0.49 at kept sites.
  - Their homozygote wrong-allele rate is 4-5x that of kept sites.
  - At matched depth they fit the founder paintings far worse: 0.93 against 0.06 nats per sample for
    depth-heterogeneity-only removals on chr3.

  There is no 2x excess-depth (duplication) mode. The removed sites form broad deficit-depth shoulders, consistent
  with cross-species reference divergence and mismapping. Two relaxations of the depth rules, one pooled and one
  per cell, failed a depth-matched thinning control and admitted severe high-depth inconsistencies. The filter's
  rules and decisions are unchanged.
- The heterozygote-balance rule (r4) never applies at TroMau coverage. Its clean class is exactly binomial, and
  its separation gate (three binomial SDs at the median pooled heterozygote depth) depends on coverage. A
  replacement was built and validated: a clean beta-binomial with fitted overdispersion against an artefact
  beta-binomial, separated by three intrinsic SDs. It would remove a further 2-4% of today's kept sites, and those
  sites fit the paintings about 3x worse than the sites retained.

  The replacement is deferred. On TroMau chr13 it removes the few sites that distinguish two full siblings. The
  siblings then become identical in family phase's discrete coordinate keys while their likelihoods differ, and
  family phase does not yet support such connected duplicate samples: it stops rather than order them by name.

### Discovery sites and calibration failures
- Discovery uses a fixed minimum of five total reads per site across the analysed cohort. The threshold no longer
  grows with sample count (formerly max(5, 0.02 N)), and later inference still uses the other sites. No discovery
  site or block changes on TroMau, std_n80, stress_n80, the N=320 N-series or sim_s10000.
- A read-model calibration failure no longer stops the run:
  - a shared-binomial polish failure discards only the failed start;
  - distinct successful endpoints are resolved by training likelihood only when the best leads every distinct
    competitor by more than one nat;
  - otherwise that contig uses the fixed binomial model;
  - decisions are logged ("[Read model WARNING]") and kept in the evidence diagnostics and run summary;
  - ordinary empty-fold fallbacks remain unflagged.

### Numerical record
- E3's 2.3 shared-curvature change of at most 1.97e-11 (absolute) in applied GLs, with every selected model
  unchanged, is accepted and documented.
- After the 2.1 Newton polish, GLs move at most 3.67749e-10 (relative) under a changed numerical environment across
  all 22 TroMau contigs. This replaces the stale "about 0.3 nats" limitation.

### Memory and speed
Results are unchanged.
- The VCF/BCF decoder writes straight into the final shared count array: 32.66 -> 8.25 bytes per input cell on a
  10,000-sample check.
- Input-site counts are released after site selection.
- Likelihood tables are built per active sample, and the observation mask needs no full-size temporaries.
- Record compression scratch is bounded and reserved.
- `founder_view()` shares the immutable counts and likelihoods and copies only the observation mask (1 byte per
  cell instead of 33). Local selection neutralises its private block copy, so its persisted search keys are
  unchanged.
- The site filter processes bounded site tiles with sparse sample-QC cells and reserves its real scratch. Its
  scratch is 0.54 GB at N=10,000 on a chr3-sized contig, against about 300 GB for the former whole-contig float
  arrays.
- Peaks on TroMau chr3 (N=116): site filter 4.05 -> 1.05 GiB; decode 3.08 -> 1.53 GB; read model 1.80 -> 0.49 GB;
  likelihoods 2.19 -> 0.85 GB; founder view 2.03 -> 0.92 GB.
- At N=10,000 the remaining evidence peak is dominated by the input counts and their one sample-order copy (about
  120 GB on a chr3-sized contig), down from about 0.5 TB.
- Evidence is also faster. On 76 cores the site filter with sample QC takes 8.0 s on chr3 and 9.2 s on chr7,
  against 20.0 s and 22.9 s in 2.3. Site work runs as one shared queue through the run's dynamic CPU budget,
  largest spans first, so freed cores pass to the remaining work. Mixture likelihoods are compiled and
  parallel, and fixed genotype likelihoods are cached per site.
- Decode plus site selection takes 11.6 s against 26.5 s on chr3, and 12.2 s against 28.0 s on chr7.
- Average core use in the filter is about 21%. The rest is the fits' sequential optimiser steps; in a full run
  evidence overlaps with discovery.
- Across all 22 TroMau chromosomes the largest per-chromosome evidence peak falls from 4.54 to 1.42 GiB.

### Validation
- Focused tests: 15 for the failure policy, 5 for the tiled filter, and the memory checks, including exact
  likelihood lookups and founder-view sharing.
- Evidence products identical to 2.3 on all 22 TroMau chromosomes and on std_n80 chr1/chr2.
- Downstream results identical on TroMau chr19 through discovery, local selection, assembly, painting and founder
  refit (386,368 values).
- A 3-chromosome TroMau subset completes end to end.
- No simulations were rerun: the site filter is off for simulations, and the discovery-site census shows no change.

## 2.3.0 (20 October 2026)

### Scientific model and released results

- Apply a **genome-wide consistent pedigree M1 floor**, fixed before calibration and resampling. Retain every eligible parent whose total raw M1 is at least the best eligible total minus 60 times the existing information temperature of the total markers. Retained parents keep raw chromosome scores; other identities share the exact-sum chromosome floor profile. Every eligible identity, contamination term and prior denominator remains. This changes the evidence model and can change inferred panels.
- Build M2 from **every eligible pair of retained parents**, including all cutoff ties. No top-20/40 truncation or sample-index tie break selects a prefix. The M2 likelihood model is unchanged. Excluding individually weak M1 parents can still exclude a strong raw M2 pair; this admission assumption is explicit. Missing-parent component fallbacks use the original raw M1 factors.
- Use a certified selective M1 screen by default. Every candidate whose bound cannot establish its result receives exact HMM scoring. Separate compact-profile and fixed-score checkpoints prevent a second floor transform. Four-dataset validation preserves all numerical pedigree/release fields, 4,060 bootstrap/LOCO records and 60 CSV files against the adopted exact model. Dense downstream arrays remain.
- Speed up full-marker pedigree exclusion with lower bounds from completed windows and exact completion whenever exclusion is not certified at the original genome-average threshold. Biological calls and observed-site counts are unchanged. Two nonempty diagnostic tables expose certified intervals explicitly; zero-target tables retain their original columns. Four-dataset report checks pass (TroMau/std actual G8 artifacts; stress/sim zero-target G7 carry-over). TroMau avoids 26.224% of pair-site visits: kernel CPU falls from 20.955 to 19.741 s (**5.8%**). This is a kernel gain, not a whole-stage or scaling result.
- Represent tied A-step assignments fractionally and carry complete tied discovery/selection families. Expand complete parent-score classes, rank replacement starts by their parent class, use ordinary/escalated replacement budgets of 48/96 (escalation off by default), and screen tied nearest-row/consensus merges without an index convention. Biological priors, state likelihoods and the binned switching objective remain unchanged.
- Enable 256-landmark discovery proposal geometry by default; explicit zero retains the dense reference. All observations remain in consensus, fitting and scoring, and whole over-budget classes are reported as search omissions. This is an approximation: latent panels/search paths may change even when released calls agree. Residual activation depends on the number of fractional records, not only sample count. All 38 approved accuracy cells retain their seven truth metrics; 56 invariance controls pass. These component results do not replace the final pipeline matrix.
- Finish shared-binomial read calibration at stationary endpoints with stable, content-ordered reductions. The E3 curvature and box-constrained Newton fix preserves the objective, bounds, all-start policy and standard calibration choices in 110/110 contig comparisons; the accepted maximum applied-GL residual is **1.97e-11**. Earlier stationarity changes selected three sim_s10001 models, so the broader read-model change is not described as universally output-identical.
- Family phase uses sample-name-independent coordinates, continued endpoint families and explicit display phase breaks where relative orientation is unsupported or MAP and polish disagree. These breaks preserve point genotypes and recombination selector chains; they do not claim to repair long-range orientation. The proposed **C8 orientation estimator is deferred to 2.4**.
- Integrate the exact Y sparse-support kernel and stage6ab first-pass sharing, scratch-before-CPU admission, initial work-based chromosome leases and dynamic load balancing. Eight std/stress single-contig reference/reversal/rename/noise controls each preserve all 22 released hashes and every truth/phase-set metric. Whole-stage AB timing remains separately measured.
- `haplotypes evaluate` now scores released phase sets exactly as exported. Explicit/component boundaries split phase sets; displayed-undetermined calls contribute unordered errors without splitting their phased flanks. Summaries identify `export_phase_sets_v2`. Baseline comparisons must use this same metric.
- Recombination CSR edge grouping and integer bincount totals, exact pedigree pair counting/SCCs, calibration aggregation, and compiled exact residual-clique counting preserve scientific results. Process and in-process CPU admission retain active claims until workers release them, including under memory pacing.

### Accuracy interpretation

The K30 forward-only baseline was convention-dependent. On the common set of 2,946,890 calls, 2.1 forward/reverse have 2,943/3,424 called errors and 224,959/229,273 wrong-or-missing genotypes; 2.2 has 3,159/3,396 and 226,449/230,891. Discovery candidate9c has 3,227 and 227,113: +43.5 common-call errors and -3 wrong-or-missing against the 2.1 convention mean, and -50.5/-1,557 against the 2.2 mean. These are component-stage descriptive comparisons, not proof of final-pipeline equivalence. The accepted assembly costs below are reported separately.

### Assembly and founder release

Completion's held-out folds now use exact integer total-depth vectors to distinguish
samples with the same observation mask. Identical depth groups stay whole and are
balanced deterministically between folds. The split uses no sample names, input index
or genotype-likelihood value, so sample permutation, renaming, convention reversal and
GL-only perturbations preserve fold membership. If a fold has no informative training
observations, its learned panel remains the original pre-fill panel and the absence of
held-out training is recorded. Original calls, cavity-fill and occupancy rules are
preserved. This corrects input-order-dependent completion fills.

Assembly now evaluates complete eligible join proposal classes and scores a proposed
founder panel independently of the edit used to obtain it. Tied join and count-choice
panels can continue into larger contexts, with common sample/site masks and same-level
refinement. A shared allowance of sixteen incoming/search leaves bounds higher-level
work; ordinary isolated join searches retain four. A complete encountered family is
evaluated or retained with explicit verification flags. Count candidates evaluate the
complete observed endpoint family before applying the existing acceptance rules.

Computationally incomplete count comparisons retain the incumbent decision and record
`count_unverified`, including cause, scope and domain size. They never justify a founder
deletion or withheld call. Complete nearest-shadow enumeration has a separate budget of
2048 combinations; exact conservative proofs may verify larger families. Such a proof
must hold for every sample permutation of the original fixed bootstrap draws. A
proof refusal preserves the incumbent and its verification flag. Evaluated
conflicting count verdicts retain the stated smaller-K and exact minimum-switch priors.

Founder output now records unsupported assembly continuity explicitly. Founder VCF
columns and tracks split into release segments when evaluated alternatives disagree
beyond a whole-panel row permutation. When invariant flanks establish a unique exact
row correspondence, agreeing allele cells are retained and only differing or missing
local content is withheld. For unequal-K terminal alternatives, a full-span path
occurring exactly once in every evaluated member also retains its known prepared
alleles. Unmatched duplicate rows use a conservative multiplicity rule. Unperformed continuations preserve incumbent calls and retain their evaluated
source families as `continuation_unverified`; that flag alone causes no masking.
Evaluated budget-overflow ties can still withhold their observed differing cells where
exact local provenance establishes the disagreement.

Exact bootstrap reuse and a bounded optional cache for identical refinement windows
reduce repeated work without changing scientific decisions. Cache omission and eviction
change computational work only.


#### Assembly accuracy trade-offs against 2.2

These are missing-inclusive founder allele-error deltas (point panel / released
panel), not offspring genotype or phase-error deltas. These three costs are accepted for the release. No rule or parameter was tuned to truth.

- **f3_fam: +107 / +107.** Completing the seven incoming branches under JOINT16
  lets the larger-context data choose the panel. The unique L4 join winner improves
  the join likelihood by 9.47 nats, with a 7.91-nat runner-up gap; the subsequent
  refinement objective ties. The aggregate delta comprises chr1 +108 and chr13 −1.
  This replaces a convention-dependent choice; it is an accepted accuracy cost,
  not a truth improvement.
- **K30: +26 / +26**, all on m2. Complete proposal-class evaluation replaces the
  earlier restricted winner. At the first divergence, the 73-proposal class wins
  by 0.330918 nats over the old 32-proposal class's winner (retained as runner-up).
  This is a data-decided trajectory change, not an overflow deletion.
- **Backcross: 0 / +51.** The point panel is unchanged in missing-inclusive truth;
  the released result contains 174 withheld cells and five continuity breaks.
  Withholding represents evaluated local disagreement with established correspondence.
  It is counted as missing in the reported errors, not hidden from evaluation.

Both dense simulations, std, stress, reallike6 and K20 have **0 / 0** deltas.
The dense missing-inclusive error totals are 1,108 (sim_s10000) and 37 (sim_s10001),
unchanged from 2.2. All eleven datasets' retained and withheld fractions, including
real datasets without independent founder truth, are reported in the assembly validation record.

#### Count verification and computational limits

Keep complete-shadow enumeration at **2,048 combinations** and the shared higher-level
incoming/search allowance at **16 leaves**, never 16 × 4. The corrected proof verifies
**1,501/1,549 decisions (96.9012%) on fixed14** and **1,246/1,284 (97.0405%) on TroMau22**.
These cover complete count criteria, not merely the fraction of candidate families
whose size fits the budget. This coverage is accepted for the release.

Final7's earlier 99% figures (99.0316% / 99.5327%) used a weaker, sample-order-dependent
proof. Final8 requires the stronger permutation-safe proof; it is never weakened to
meet a coverage target. On the observed domains, reaching 99% with the corrected proof
would require budgets of **16,777,216** (fixed14) and **7,077,888** (TroMau22). These are
domain projections, not runtime-validated settings, and are not adopted.

When a complete family exceeds the budget and cannot be proved, evaluate no prefix:
hold the incumbent and record `count_unverified` with cause and scope. Computational
unknowns never justify deletion or masking. Unperformed continuations similarly retain
calls and evaluated source records under `continuation_unverified`. Withholding is
reserved for established disagreement among evaluated alternatives.

### Measured resources, targets and remaining validation

- TroMau v23c4 completed on 112-core **AMD EPYC 7763**: pipeline 2,947.967 s (49.13 min), complete acceptance task 3,011.076 s (50.18 min), sampled summed process-tree RSS 62.524 GiB. The raw wall ratio to the 2.1 Sapphire reference (2,298.5 s) is 1.283; hardware differs, so this is not a software-only ratio. Summed RSS is not comparable to the historical 41.74-GiB peak-process RSS. Later exact G7/C1/G8 swaps carry biological results by their documented identity evidence, not these runtime measurements.
- The adopted G7 screen on matched 112-core EPYC simulation inputs takes 121.286 s / 3,480.03 CPU-s versus 322.982 s / 29,833.15 CPU-s exact: 2.663x wall and 8.573x CPU speedup, with 96.802% fewer exact chromosome-dyad queries.
- The whole family-phase target is about **1,200 s at N=320 on 112 cores**, originally specified for Sapphire. The latest completed Y measurement on EPYC is **1,492.322 s** for the family stage plus **321.142 s** post-fit (1,813.464 s combined), 110,610.43 contig CPU-s, 74.12 mean busy cores, 469.605-GiB sampled whole-tree peak and 83.560-GiB largest task. AB's final whole-stage measurement on the same EPYC 112-core host is **1,570.299 s** for the family stage plus 345.638 s post-fit, 110,665.76 contig CPU-s (ideal CPU/112 988.1 s), 70.47 mean busy cores, a 343.2-GiB sampled peak and 87.99-GiB largest task, with chr3/chr7 22/22 hash-identical and zero truth deltas. It was about 5% slower than Y in this run because fewer contig workers ran concurrently; the target is not met on EPYC. Available node memory and normal headroom govern concurrency, not the historical 1.5x proxy cap.
- The fixed-genome scaling target remains about **N^1.2** outside pedigree. The final N-series measures N80/N320 on the same frozen **v23c5 (G7 plus C1X)** source and EPYC hardware; N80 is 327.576 s. **N320 did not complete within its 45-minute cap:** discovery of a single chromosome took 1,564.7 s, with the search incomplete in 575 of 580 blocks, so no end-to-end exponent is measured. Discovery on this design is the open scaling issue. N160/N640 are unmeasured unless separately reported. A two-point slope is descriptive, not an asymptotic proof, and does not measure the later AB speedup.
- Pedigree's restated target is linear expensive likelihood/decision/resampling work with bounded plausible parents, a certified all-pairs bound projected at **at most 10% of whole-pipeline runtime at N=10,000**, and no dense quadratic default-path arrays beyond compact screen storage. **This target is not met:** dense downstream arrays persist. The full-genome bound cost 18.3669 ms CPU/pair projects to about 4.56 hours at N=10,000 under ideal 112-core utilization; a matched-genome whole-pipeline denominator is required before claiming a fraction. Do not divide that full-genome cost by the five-short-chromosome N-series fit.
- Final integrated truth/noise/reversal/rename coverage is incomplete. TroMau completion is not a truth or invariance pass: versus 2.2 it has 33,301 changed called genotypes, 118,698 call/unknown differences, 13,942 relative phase switches, and one changed parent set; these cross-version differences require attribution, not zero-difference acceptance. Actm3 does not carry from c3: actual residual pools reach 310, above 256, and no c4 reference has completed. Final-tree actm3 therefore rests on component evidence only. The std reference has a complete 11,822-block inactivity census (maximum residual pool 184); its reference biological outputs carry through the documented component identity chain, not through a fresh final-tree run. This does not validate its noise/reversal/renaming controls. Every deadline-skipped cell is reported as NOT_RUN, independently of pool exit status. Full final sim_s10000 truth, the remaining truth panel, broader zero loop, TroMau reversal and final Gate W remain follow-up where uncompleted; component gates remain explicitly labelled.

#### Evidence freeze and uncompleted cells

Evidence collected at **2026-10-20T08:22:56.652274+01:00**. At the evidence freeze (2026-10-20T08:22:56.652274+01:00), the exact terminal NOT_RUN list is empty: no deadline guard has yet skipped a task. Pending or running cells are uncompleted, not passes. The final report must add any later NOT_RUN records without changing this frozen transfer.

Uncompleted final truth references: stress_n80, lad_K20_n80, lad_K30_n80, f3_fam,
backcross_s6202, reallike6_nf_s3, actm3 and sim_s10000. The std reference relies on
the qualified carry-over above; its queued c4 reference has not started. All six
queued std/stress controls (iid12_s1, reversal and rename_s1 for each) are uncompleted.
N320 did not complete (discovery time cap; see above); N160/N640 are unmeasured. AB whole-stage timing: 1,570.299 s family stage (see above).
TroMau reversal, the broader zero loop and final Gate W also remain uncompleted.

### Follow-up work

Pedigree sparse downstream decisions/exposure/cavity/path/resampling, complete N-series and omitted release validations remain open. SIMD vectorisation may use the owner's post-2.3 few-ULP policy (reassociated reductions/FMA only when released decisions are unchanged); this policy is not applied retroactively to 2.3's exact-swap gates. C8 orientation estimation continues for 2.4. G8 is included after its exact-exclusion, interval-report and default-handler/checkpoint gates; its modest kernel saving does not resolve the remaining pedigree scaling requirement.

## 2.2.0 (19 October 2026)

Release 2.2 removes arbitrary conventions from every decision that selects between different biological answers.
Tied alternatives that more data can separate are carried forward and decided by the data. Ambiguity the data cannot
resolve is released as such: unknown alleles, switch intervals, phase-set breaks, unordered parent slots, and flagged
alternative sets. The remaining exact ties are decided by stated priors ([docs/methods/ties.md](docs/methods/ties.md)).

Validation of the release candidate (19 October):

- **Noise loop.** Released outputs were identical in every cell. Compared were founder alleles (unknowns
  included), founder counts, painting (intervals included), pedigree, phased calls, crossovers and the VCFs. The
  cells were 1e-12 and 1e-9 relative genotype-likelihood noise, value-consistent noise, permuted sample order (in
  memory and in the input file), another thread count and another numerical environment. They covered three
  TroMau chromosomes, std_n80 and stress_n80, and the TroMau whole genome under 1e-9 noise and permuted order.
- **Convention reversal.** Released outputs change only through the conventions listed in
  [docs/methods/ties.md](docs/methods/ties.md#remaining-conventions):
  - stress_n80: no genotype changes; 13 phase switches; crossover intervals of 26 meioses.
  - std_n80: 6 genotypes; 20 phase switches; 20 meioses.
  - Three TroMau chromosomes: 1,361 genotypes and 1,713 phase switches, from 26 discovery blocks.
  - TroMau whole genome: 22,827 genotypes and 11,056 phase switches, from 326 of 9,979 discovery blocks (2.1
    with the reversal switches: 46,758 and 20,021). One Tier-B status changes.
- **Truth against 2.1.**
  - Pedigree calls are identical on every simulation evaluated, with one more exact configuration on the K = 30
    ladder. sim_s10000 was still running at release.
  - stress_n80: genotype errors -20%; allele errors per released phase set -39%, with 137 more phase sets (N50
    -0.6%); phase switch errors 98 -> 94.
  - std_n80: allele errors per phase set +19%, all from one sample's orientation at a painting tie boundary;
    genotype errors -148.
  - f3_fam: allele errors -6.6%; genotype errors +7%.
  - reallike6: allele errors -8%.
  - backcross: allele errors per phase set -9%, with 33% more phase sets; genotype errors +0.3%.
  - sim_s10001: genotype errors +4 of 2.87 billion calls; allele errors per phase set -15%, with 7,040 -> 11,168
    phase sets (N50 -5%); 3.3% of called alleles are released unphased where the frame is undetermined.
  - K = 20 ladder: identical.
  - K = 30 ladder: genotype errors +9% and phase switch errors 170 -> 216, mostly the cost of the order-free
    A-step rule. The final panels recover the same 143 of 150 true founders as 2.0 and 2.1.
  - On a metric that splits each sample at the painting tie boundaries of both releases, stress_n80 is 19% worse.
    That comes from long orientation errors in four two-parent samples; f3_fam and reallike6 are better on it.
- **Resources.** The TroMau whole genome took 37:12 against 2.1's 38:23, with peak memory 42.3 GiB against
  41.7 GiB. Dense simulations take up to 1.13x 2.1's wall time and 1.06x its memory.

### Declared model changes

- **Discovery: tie-carrying residual clustering.** Every tie resolution of the residual clustering is carried as a
  candidate for the block score to decide: level-wise HDBSCAN* condensed-tree groups and maximal-clique alternatives
  of the noise, with order-invariant bounds. It replaces 2.1's content-key order of equal-weight links and the
  complete linkage on noise. Archive beams keep whole NLL classes. Emission, deduplication, K = 1 seeds and the
  replacement and gauge budgets are order-invariant. Founder alleles that no carrier observes are released unknown
  (`latent_unknown`). NLL-tied panels are released with the block (`tied_panels`).
- **Discovery: tied A-step states take their meet.** A sample whose tie band holds several states (within the
  fewest-unknown-copies category, a kept prior) is assigned the multiset intersection of the founder pairs; a
  copy that differs between the tied states goes to no founder. This depends only on the tied set, not on row or
  sample order, and replaces 2.1's row-order rule in discovery. The objective is unchanged: the unknown-copy
  count stays the tied states', and the cavity's state prior counts a tied sample 1/|T| at each tied state.
  Accepted cost on the K = 30 ladder: discovery misses 6 more true founders, summed over blocks, than with the
  row-order rule (10 fewer than 2.0, 3 more than 2.1). After assembly every release recovers the same 143 of 150
  true founders, but genotype errors rise by 6% and phase switch errors from 170 to 216. A fractional treatment
  is the planned next step.
- **Local selection: tied sets decided by the data.**
  - Every source's cavity tie set is fitted and searched, instead of 2.1's last-source start.
  - The BIC tie set of the bank is kept whole, and tied donor completions each give a source
    (`CandidateBankConfig.completion_variant_cap` = 8).
  - Whole tie classes are refitted (`PathSearchConfig.refits_per_kind_cap` = 16).
  - Residual ties give a consensus panel that withholds rows not shared by every alternative (`tied_alternatives`,
    `withheld_by_ties`).
  - Discovery's tied panels become starts and bank members.
- **Assembly.**
  - Batching is orientation-symmetric.
  - Inside log-likelihood tie classes the batch linker's path score decides join moves.
  - Painting-dependent proposals run under both traceback conventions and are merged.
  - Tied count-step refits that disagree on the founder count take the smaller count (prior); otherwise the member
    whose margin most strongly supports the decision wins.
  - The two fixed-K refinement searches use opposite conventions and keep the better F. Equal-F alternatives are
    flagged (`ambiguous_k`, `alternatives`).
  - Founder refit releases mutually exclusive tied moves as unknown cells.
- **Painting: switch intervals.** A switch between founders the sample cannot tell apart is released as an interval
  (`SWITCH_INTERVAL`) instead of a point at the leftmost tied edge. Tracks next to an interval are released only when
  2.1's gate passes under both traceback conventions, which is never more permissive than 2.1. Founder alleles inside
  interval bins are those both candidates share. On stress_n80 this removes 366k calls, of which 6.4% were wrong
  (genotype errors 117,369 -> 93,909), at a cost of 11 more phase switch errors (98 -> 109). The per-sample segments
  are published (`painting/<contig>/sample_segments.tsv`).
- **Pedigree: parent slots.** Parent1/Parent2 follow recorded sexes (dam, sire). Without recorded sexes the slots are
  unordered (`ParentSlotBasis`).
- **Family phase: a schedule-independent family MAP and released undetermined frames.**
  - The ascent runs under the canonical schedule and under its mirror. Difference clusters are transplanted,
    best exact gain first, into the end point with the higher posterior, and the full ascent then continues. It
    stays monotone with exact gains.
  - New moves:
    - compound moves with terminal segments;
    - a joint sibship genotype refit inside compound seeds (at most 64 joint configurations per site, whole sites
      skipped, never truncated);
    - sibship pairs with tolerance;
    - co-located pairs (siblings switching different parents' bits at the same markers).
  - The long compound tier is off by default (`FamilyPhaseConfig.map_long_moves`).
  - `frame_undetermined` marks sites where another tied configuration sets the frame differently (bit 0
    decoder, bit 1 displayed), and the export breaks the phase set there (`phase_breaks`).
  - One-parent children are shown in parental origin: the painting phase inside each stretch between its painting
    tie boundaries, each stretch oriented by its summed family-MAP evidence.
  - Root-frame repair votes: a child opposes a change only if its exact gains agree with its flanks. This removes
    both 2.1 transfers on stress_n80, including chr10 F1_8 (+43k allele errors in 2.1).
  - Recombination decodes the non-homozygous sites of an undetermined decoder frame as unknown, so crossovers do
    not depend on where a tied switch is placed.
- **Export.** Founder alleles with no observing painted carrier are written `.`, counted in the manifest and the
  founder VCF header (`unobserved_founder_alleles`). Phase sets break where family phase leaves the displayed
  orientation undetermined.

### Convention reversal and evaluation

- `HAPLOTYPES_TIE_REVERSE=1` reverses every remaining arbitrary convention: index, content-key, default-state,
  sample-name and keep-first orders (`maths.ties.reverse_conventions`). It is the release acceptance test.
  `HAPLOTYPES_TIE_REVERSE_KEEP` keeps named discovery conventions at their default for attribution. Records written
  under reversal carry `tie_reverse` in their identities.
- The evaluation reports allele errors aligned per released phase set (`ps_aligned_allele_errors`), the number of
  phase sets with at least two called heterozygotes (`phase_sets`), their N50 (`phase_set_n50_bp`) and the called
  alleles of undetermined frames. The component-aligned metric is kept for continuity, but it charges a released
  phase-set break as orientation errors.
- The run report gains `painting/<contig>/sample_segments.tsv`, and `ties/summary.json` records whether conventions
  were reversed.

## 2.1.0 (19 October 2026)

Release 2.1 makes the results independent of floating-point rounding, of the input sample order, of the thread count
and of arbitrary tie-breaks, replaces family phase's loopy belief propagation with a convergent solver, and searches
discovery more broadly. On the release candidate of 18 October, the noise-stability matrix found zero changed outcomes
at every stage, from evidence to export. The matrix covered 1e-12 and 1e-9 relative genotype-likelihood noise,
value-consistent noise, permuted sample order (in memory and in the input file), a changed numerical environment
and a different thread count, on simulated crosses and the TroMau benchmark. Against 2.0, that candidate gave:

- identical pedigrees on every simulation, and one more correct edge on the K = 30 ladder;
- phase switch errors and component-aligned allele errors within a few percent either way (allele errors -22% to
  +6.5% by simulation);
- on the TroMau benchmark, slightly lower founder coverage (best-row coverage -0.012) and fewer painted F2
  switches;
- TroMau wall time within +6% of 2.0, with a peak memory of 62 GB against 44-59 GB.

The release adds the family-phase solver and late fixes to that candidate. It was checked end to end before
release: three TroMau chromosomes and the std_n80 and stress_n80 simulations completed cleanly. Their pedigree calls
equal the candidate's, and so do std_n80's genotype, phase-switch and allele errors. On stress_n80, allele errors
are +1.6% from the new family-phase solver. The full noise matrix was not rerun on the final code.

### Declared model changes

- **Local selection, context code: claims coded by the cheapest matching.** Context founders claim panel rows by a
  maximum-cardinality, then minimum-edit-cost matching of founders to their nearest rows (the panel's shortest
  description), so the code length is a function of the panel's rows whatever their order. 2.0's greedy rule (a
  founder equidistant to several rows claimed the lowest; applied to a trajectory-dependent row order, it let the
  same row set cost up to ~140 nats more or less) is removed.
- **Founder refinement: zero-gain founders are deleted first.** A founder whose removal leaves every sample's score
  unchanged (within the tie quantum) is removed before each deletion round, which can lower the founder count where 2.0
  kept a founder no sample uses.
- **Discovery: broader search.** More data-birth attempts and children (x3), beam width 6, and tie groups of
  equally ranked data births consumed whole. The founder-count criterion is unchanged; the broader search finds
  better-optimised lower-K panels more often, which lowers K in some blocks (accepted).
- **Family phase: loopy belief propagation is replaced by a provably convergent posterior-mode solver** (the family
  MAP). It runs monotone exact block coordinate ascent, with pedigree roots' genotypes summed out. Every accepted
  step raises the posterior by more than a tie quantum, so the solver terminates; reaching the sweep cap
  (`FamilyPhaseConfig.map_max_sweeps`) is an error. Decoder frames, root-frame repairs and crossovers can change
  where 2.0's propagation had not converged. Known limitation: the solver can stop at a local optimum; larger
  moves are planned for 2.2. `family_phase/summary.csv` reports `map_sweeps`, `converged` and `map_objective`
  instead of `iterations`, `converged` and `final_residual`.
- **Read model: fits polished to their exact optimum**, so genotype likelihoods do not depend on the numerical
  environment (BLAS kernels, SIMD paths).

### Noise stability

- One canonical sample order per run (samples sorted by name) from the evidence stage on; exported VCF columns and
  per-sample report tables keep the input order.
- Tie classes with per-kind tie quanta (`maths/ties.py`) replace exact floating-point comparisons and array-order
  tie-breaks in discovery, local selection, assembly, painting, pedigree, family phase and recombination; ties are
  decided by exact content keys and every panel is kept in canonical row order. Threshold comparisons get a band
  only where a value can sit exactly on the threshold mathematically.
- Results no longer depend on the CPU's vector instructions. Discovery's HDBSCAN (K = 1 seeds, residual-strand
  clusters) orders links of equal weight by a fixed pseudo-random key of the canonical record indices; the hdbscan
  package's unstable sort of equal-weight edges differed between AVX-512 and other CPUs. This equal-weight-link
  rule is a content-key convention under review. Logarithms and exponentials in discovery and the block scale go
  through scalar libm, and Numba's Intel SVML is disabled (`NUMBA_DISABLE_INTEL_SVML=1` at package import).
- Local selection decides ties without index or hash order. Bank winners inside wide tie classes are ordered by
  (K, canonical key). Proposals tied with the eighth refit of a kind are refitted too, up to
  `PathSearchConfig.refits_per_kind_cap` = 12. A larger MAP unknown-copy count must win by more than
  `PathReleaseConfig.unknown_count_band` (2^-20) of the largest count mass, else the smaller count stands.
- Founder refinement's exchange and content moves, its coarse-move guard and founder refit's interacting allele
  moves are decided by tie classes of their exact gains, then content. Founder refit judges each site's moves
  against the round-start state; it is identical to 2.0 when candidate moves do not interact.
- Ties are decided by documented deterministic conventions and counted. Where alternatives tie (exactly equal
  support, or within one tie quantum) a fixed rule decides: content keys, a parsimony preference (smaller panel,
  fewer parents, fewest switches), keeping the current state, sample-name order, a conservative threshold side,
  or abstention. [docs/methods/ties.md](docs/methods/ties.md) lists them per stage. They are conventions, not
  inferences, and are under review for 2.2. A completed run writes `ties/decisions.csv` and `ties/summary.json`
  (decisions taken inside a tie per contig, stage and decision kind; discovery blocks selected inside a tie,
  their tied alternatives and score margins; ambiguous founder-count calls; tied refits that disagree) and a
  one-line digest in the log; the fields they come from (`tie_events`, `ambiguous`, `tied_alternatives`,
  `score_margin`) are documented there. The plan for 2.2 is to carry every tie resolution forward as a
  candidate and let the block score decide.
- Fixed: an assembly join could cycle forever between a birth, a death and a seed swap (count-state revisit guard
  and a reported backstop).
- New diagnostic environment variable `HAPLOTYPES_TIE_CENSUS` (valley census of tie decisions; results do not
  depend on it).

## 2.0.0 (unreleased)

The package is rebuilt from release 1.0 (commit 92b3daf) as one pipeline of typed stage packages joined through the
checkpoint store, with one driver, one configuration and one execution layer. The science is that of 1.0: the
algorithms, models, thresholds, release and tie rules are unchanged, apart from the fixes below. Accepted on
18 October 2026 against 1.0 in the same environment on nine reference scenarios
([validation](docs/validation.md#results)): every difference is attributed to a documented fix or to rounding and
ties, the pedigree truth metrics are identical on all seven simulations, the whole-genome simulations run 3-5% faster
with a 27-34% smaller largest process, and TroMau takes the same wall time (41:10 against 41:00) with a 16% lower
node peak. [Migration](docs/migration.md) maps every old command, setting and output name to the new one.

### Incompatible changes

- One command, `haplotypes run`, replaces `simulate`, `reconstruct`, `tropheops` and `astcal`; the crosses are the
  presets `tromau` and `aulstu`. `run.py` is removed. `evaluate`, `export`, `status`, `example`, `recombination` and
  `batch-worker` take run directories and destinations as positional arguments.
- The configuration has new sections (`[execution]`, `[input]`, `[samples]`, `[quality]`, `[genetic_map]`,
  `[pedigree.eligibility]`, `[family_phase]`, `[simulation.reads]`, `[reports]`). Unknown keys and sections are errors
  (1.0 ignored them). Scientific switches are `--set SECTION.KEY=VALUE` instead of one flag each. `[input] contigs` is
  required for VCF input unless a preset lists them.
- No setting is read from environment variables any more (`HAPLOTYPES_*`, `BHD_*`, `HAPREC_COARSE_MOVE_GUARD`).
- One stage vocabulary: `block_discovery` is `discovery`, the feedback rounds are `local_selection`
  (`local_selection_initial`, `_l1`, `_l2` in evaluation), `allele_refit` is `founder_refit`, family refinement is
  `family_phase`. Output directories are named after the stages (`quality/`, `painting/`, `founder_refit/`,
  `family_phase/`, `recombination/`, `simulation/`); `pedigree/scientific_relationships.csv` (a copy of
  `tier_b_relationships.csv`) and constant diagnostics columns are removed.
- Checkpoints written by 1.0 are not read: the container format is unchanged, but record keys and identities are new.
  A stage's identity covers the source of the packages it runs (computed automatically), its configuration and the
  record identities of its inputs.
- Removed alternative modes: founder refinement always runs (`--founder-refinement off` is gone) and its coarse-move
  guard is fixed to the shared-junction guard; the pedigree direction alternatives (cluster, continuous-only and
  family-only) and their internal ablation switches are gone; the pedigree keeps one current record per product
  instead of versioned decision replays, so a changed pedigree setting needs a new output root.
- Python callers: `core/` and `workflows/` are replaced by `config/`, `store/`, `execution/`, `pipeline/` and one
  package per stage (`evidence/`, `discovery/`, `local_selection/`, `assembly/`, `painting/`, `pedigree/`,
  `founder_refit/`, `family_phase/`, `recombination/`), with `evaluation/`, `reports/`, `export/` and `templates/`.

### Input and checks

- One record policy for every VCF/BCF: records that are not biallelic SNPs are dropped and counted (`not_snp`,
  `multiallelic`), every record at a position with two or more biallelic SNP records is dropped
  (`duplicate_position`), and I/O or format errors stop the run. Real BCFs with indels are accepted. Results are
  unchanged wherever 1.0's site filter ran (its rule r0 removed the same records); 1.0's cross route truncated
  multiallelic depths to the first ALT and could swallow read errors, its generic route rejected any file with an
  indel, and runs without the site filter read indels as SNPs.
- Everything that can be checked runs before any computation: configuration consistency, the variant header (AD,
  samples, contigs present and populated), the workbook, sheet, columns and sample list, and the eligibility policy
  (1.0 read the cross workbooks only after painting). Runs reaching the genome-wide stages need at least three
  contigs; a run with fewer than five contigs warns at the start that Tier-B calls are unreachable, and the pedigree
  states its count of informative chromosomes and whether each tier was reachable.
- Genotype likelihoods of every read model come from one emission kernel; values change at rounding level only.
- Read model: the shared homozygote-overdispersion candidate and its null refit are removed (homozygote kernels are
  binomial); held-out selection scores within one nat of the best are ties and the simplest candidate is applied;
  optimiser stop reasons do not decide which fit is used (heldout-read-model-v6).
- A run marks its checkpoint directory with the release layout before writing anything and refuses a directory
  holding checkpoints without the mark (an earlier release) or with other subdirectories. A shard of a simulation is
  refused until the simulated inputs of every contig are recorded.

### Scientific fixes

Each was measured on the bounded references against the unfixed tree; [validation](docs/validation.md) has the
acceptance evidence.

- Discovery: a cycling fixed-K fit returns the cycle's lowest-NLL state, reports the iteration limit and is never
  certified as a fixed point (1.0 returned the state the iteration limit landed on, which depended on the limit's
  parity). Cyclic-fit endpoints changed; no block outcome changed on the references.
- Assembly count rule: the row geometry (nearest rows, shadows, near-identity and novelty tests) reads the pre-fill
  calls, so a completion fill never decides a count (1.0 measured on the published calls). Founder refinement's count
  rule counts S as the component's informative samples (1.0 counted every evidence row, founder-excluded samples
  included), and a deletion round is skipped by its bound only when the bound also proves the bootstrap band (1.0
  recorded such rounds as not ambiguous without evaluating the band).
- Painting: bin-to-bin switch probabilities use Haldane's map function (1.0: min(1/2, d)); release status changed in
  a handful of track bins with posteriors within 1e-3 of the 0.9 threshold.
- Founder refit: two-row moves are only swaps of different alleles of rows painted together at the site, accepted
  at twice the per-allele prior (10 nats); 1.0 flipped any pair of called rows on a summed 5 nats.
- Family phase: the polish never switches phase inside the first painting bin of a chromosome or component (1.0 gave
  those intervals the switch probability of a bin edge).
- Pedigree: the weakest-link tie tolerance of acyclic graph selection is 512 machine epsilons of the scores the
  margins come from (1.0 took 512 epsilons of the margins themselves, below their rounding).

### Fixes and other changes

- New export stage: every complete run writes a founder VCF and the phased sample VCF per contig to
  `OUTPUT/export/` (`[export] enabled`, `[export] full_length`); `haplotypes export` writes the same files, with
  per-component founder-track TSVs as the optional product `tracks`.
- New memory governor: parallel work is admitted by estimated peak bytes and waits when memory is tight instead of
  failing; `[execution] memory_limit` (`--memory-limit`) caps it, nested work keeps headroom for its sub-tasks, and a
  new run removes the shared-memory segments of dead runs. Pacing never changes results.
- Large founder counts: local selection, discovery, assembly and painting keep their working memory per task and
  reserve calibrated estimates (lane-block path kernels, bit-packed discovery caches, shared assembly evidence,
  batched label continuity); results are identical. Founder refinement's window programmes run faster with identical
  results.
- One CPU-lease tree and one process-pool type replace the separate redistribution mechanisms; the Numba threading
  layer is OpenMP explicitly (the `tbb` dependency is dropped).
- The resolved configuration is written to `OUTPUT/config.toml` and into every run record; stage timings include
  helper threads.
- Discovery's checkpoint chunk counter advances only after a successful write, so a failed write can no longer make
  an incomplete contig look complete. Maximum cuts above `DiscoveryConfig.search.exact_cut_max_k = 12` founders are
  reported as heuristic cuts instead of a search limit (diagnostics only).
- Assembly levels continue past L4 while a boundary that is not a phase break remains (no change at the reference
  sizes); final and context assemblies are distinguished explicitly; joined batches are keyed by their typed inputs,
  so identical joins are reused across a contig's assemblies more often.
- Founder refinement's window frontier has a fixed cap (`AssemblyConfig.refinement.frontier_bytes`, 128 GiB) with
  each fallback reported, instead of an error above a fixed entry count; sweep histories are stored once per fit.
- The painting stage owns its configuration and release rule, computed once when it is written; its memory budget
  reads the job's memory (cgroup-aware) rather than the node's.
- Pedigree identities are per contig, the evidence calibration comes from the configuration, memory planning reads
  the records instead of a published table, and the two roles of the former parent-state contamination constant are
  separate fields (same values). `OUTPUT/pedigree/summary.json` publishes the informative contigs, the
  chromosome-count statement (whether Tier A and Tier B were reachable), the family solve and the exclusion and
  eligibility summaries.
- Founder refit binds its identity before reusing a recorded contig, as every other stage does.
- Recombination's shared-orientation fit no longer fails on genetic maps with zero-rate stretches.
- Family phase releases after at most 25 belief-propagation iterations (earlier when converged). 1.0's stability
  checks after 20 iterations could not fail under the scaffold polish, so outputs are unchanged; the retry ladder,
  the 520-iteration limit and the switches restoring older behaviour are gone. Protected intervals
  (`[family_phase] protected_intervals`) and order breakpoints share one reader that requires the header line; the
  tables' content, not their path, enters the identities.
- The reference-consistency report uses the run's calibrated genotype likelihoods (descriptive report only).
- `painting/<contig>/founder_counts.tsv` and `assembly_search.tsv` count each join search once, at the level where
  it ran; the genome-wide `recombination/junctions.csv` is built from the records at full precision; the simulation
  truth pedigree table is also written on resume (columns `sample`, `cohort`, `parent1`, `parent2`).
- Export takes the biallelic SNP record at positions with several records.
