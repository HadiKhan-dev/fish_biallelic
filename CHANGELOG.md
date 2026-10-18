# Changelog

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
