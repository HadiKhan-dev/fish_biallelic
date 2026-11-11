# Changelog

## 4.14.0 (11 November 2026)

This release lets one chromosome run on several nodes at once, bounds the two searches that produced multi-hour units
at high founder counts (refinement windows and discovery's follow-up moves), corrects the pedigree's direction
evidence, removes a per-phase-set pass over the whole chain from the family-phase release of samples the pedigree does
not link, and fixes a compiled-kernel symbol collision that could crash a worker or leave a sort wrong in earlier
releases. 4.14 does not read or resume 4.13 checkpoints or exports.

**Compiled kernels: colliding symbols.** Numba names every compiled function with a counter that each process starts
at one. Its own sort routines are closures of one factory, told apart only by that counter, so two worker processes
could compile different routines (an index sort and a value sort of the same array type, say) under one symbol. Both
entered the kernel cache, and a process that later loaded both ran one routine for both callers: `argsort` returned
sorted values that were then read as indices (a worker crash with signal 11, observed in the join's splice scores), or
`sort` returned its input unsorted without an error, depending on which kernel was loaded first. Whether a cache holds
such a pair is decided by chance when its kernels are first compiled. The counter now starts at a random 56-bit value
in every process and is redrawn after a fork, so symbols compiled in different processes no longer coincide. Only
4.13.0 can be affected: it is the first release whose kernels hold a value sort and an index sort of the same array
type (the merge certificates of local selection beside the index sorts of the join, the refinement and the painting);
earlier releases sort integer arrays by value and floating-point arrays by index only. In 4.13.0 about 7% of fresh
runs of a small simulation crashed (4 of 56); in the silent order the merge certificates read unsorted values, which
can change local selection's merge screen. A 4.13.0 run that compiled its kernels itself and did not crash cannot be
shown clean from its outputs: recompute results that matter with 4.14.

**One chromosome on several nodes.** Concurrent `haplotypes reconstruct CONFIG --contig C` invocations share discovery
and selection blocks, selection context refits, founder completion, assembly batches, refinement components, painting
and founder refit through non-blocking record claims. Every invocation runs the same cut and finishes through refit.
Busy tasks release CPUs and retry after five seconds, with a 30-second heartbeat. Workers release claims on
coordinator death. Scientific functions and ordered gathers remain unchanged; context refits and completion ranges
gain lossless intermediate records. All processes must use the same fixed source tree and checkpoint root; changed
code identities require fresh checkpoints. The checkpoint container now establishes the end of its data under the
append lock by reading, not from the file's size attribute: on a parallel file system the lock and the size are kept
by different services, and a process that had just been granted the lock was observed to get, for a fraction of a
millisecond, the size from before the previous holder's record, so that it overwrote that record. A size below the
indexed end no longer resets a reader's index. One invocation on one node was not affected by either.

**Excluding contigs.** `exclude_contigs` names configured contigs to leave out of the pedigree and of every later
stage. Evidence and relatedness still read every input contig; `run` does not reconstruct an excluded contig, and
keeps any records it already has. The final products of each contig set live side by side, so the set can be changed
on resume without recomputing the included contigs.

**Founder refinement: window search.** A window template is solved by the exact bounded-frontier programme while its
frontier stays within 64 kept paths. A template that exceeds it is solved by a seeded search instead of the restart
ladder, whose cost grew as K⁴ to K⁶: the search starts from the paths that single samples favour (six whole paths
and six restrictions of a path to an interval), improves each by single-unit ascent, and scores every restriction and
every single-unit change of a start from one forward and one backward pass of max-plus messages. The returned path is
rescored by the ordinary surrogate, and acceptance is unchanged. The seeded search is a heuristic above the frontier:
it carries no optimality certificate. In component replays of two known-truth simulations (6 and 30 founders) every
released panel row equals the exact programme's. On eight saved components of real chromosomes with 12 to 23 local
founders the founder count equals the exact programme's on four and differs by one or two on four, in 13 to 71 minutes
instead of 38 to 179. The focal solves of one template also share the caller's CPUs in contiguous chunks, with
identical results.

**Founder refinement: tied incoming roots.** A join can hand a component several starting panels of exactly equal
likelihood (typically two founders exchanging their row at one block). Every such root used to run a complete count
pass, so a component with four tied roots cost four passes. Now the nominal root runs the pass, and each root of the
same count follows the deletions that pass adopted: it is converged from its own start and re-converged after each
adopted deletion of the corresponding founder, and its terminal panel enters the unchanged terminal comparison, so a
tie that survives the pass is still released as a family. A root costs one fit plus one per adopted deletion instead
of a pass. This is a search-rule change: a root whose own count decisions would have differed from the nominal root's
is no longer detected (none did in the components examined). Roots of another count, roots without an unambiguous
founder correspondence, roots one of whose fits ends in tied endpoints, and all roots after a best-F restoration keep
their own pass. The component record gains `branch_replays`.

**Discovery: screened follow-up moves.** Within each founder count, deaths, gauge rewires, soft births and
replacements are ranked by their likelihood before any fitting iteration, and only the leading classes are fitted (6
deaths, 6 gauge rewires, 12 soft births and 12 replacements, in whole tie classes). Data births are fitted as before.
This is a declared restriction of the search and can change discovery panels: the omitted moves are recorded, and the
neighbourhood certificate covers the screened neighbourhood only. Frontier batches are also partitioned by
sample-vector width, which changes no result. On 60 blocks of a real cross, stratified by founder count, discovery took
3.9 core-hours instead of 19.9 and its longest block 17 CPU-minutes instead of 180; cost grew as the cube of the highest
fitted founder count instead of its fourth power. At the discovery stage of two known-truth simulations the truth
metrics agree within half a percent (one-to-one founder errors 296 against 306, and 6,256 against 6,301).

**Assembly joins.** The sample team of a batch is sized by its pair work (bins times the square of the largest local
founder count) instead of a fixed eight samples per thread, so a batch of few samples and many founders uses the CPUs
it holds; prefix scores reuse integer pair addresses; and sparse kernels use only the lanes that hold samples.
Candidates, bounds, pruning thresholds and decisions are unchanged, and saved batches reproduce their records. On
three upper-level batches of a real 52-sample chromosome with 25 to 31 founders, alone on a node, the join took 3.5 to
5.5 minutes instead of 8.6 to 10.5.

**Pedigree direction: joint transmission evidence.** Direction now scores a child's scheduled homologue as a meiotic
copy of the candidate parent's two, with the other three homologues modelled as founder mosaics. It replaces the
inherited-junction count bound, which can favour reversed directions when a parent's painted junctions all lie on the
homologue it did not transmit in a descendant-rooted founder frame. The per-bin painting mismatch probability is 0.01;
the founder-mosaic change probability theta / (T − 1) is derived. A shared unrecombined homologue gives near-neutral
evidence, with small residuals from alternative mismatch paths; trio evidence decides those pairs. Orientation log
odds, pedigree calls, release tiers, edges and downstream phase/recombination outputs may change. Saved decision
replays of a real multi-generation cross change three samples at Tier B; a 30-founder known-truth replay loses one net
true Tier-B edge (12 to 11). These are scientific changes, not a claim of real-data accuracy. Other direction
corrections, calibration, resampling, tiers, guard and the reported junction-density graph-order contrast retain their
definitions. The stored field is `transmission_log_likelihood` (nats), in chromosome score schema 2 and pedigree
product schema 4. Changed code identities refuse old stage checkpoints; old count bounds are never decoded as
likelihoods.

**Family phase: samples without a linked parent or child.** Such a sample's chromosome is released as phase sets, each
anchored at its first confidently heterozygous bin. Every phase set used to cost one forward-backward pass over the
complete chain, so a sample cost phase sets times bins times states cubed, and a chromosome whose pedigree links no
sample became the longest unit of a run (over five hours of phase on a whole node for one 52-sample chromosome with 32
founder states). The anchor restricts the state at one bin, which changes neither the forward messages before that bin
nor the backward messages at and after it. The release now computes the unrestricted messages once and, for each phase
set, only what the set reads: the backward messages down to the set's start and the forward messages until the set
ends, taken in pieces with the continuation test after each (the last piece may run past the set's end). These are the
same operations in the same order as the complete pass, so the posteriors are the same numbers (a test compares them
bit for bit) and every released call, phase and set is unchanged; a sample's message computation is now linear in the
chromosome's length, whatever the number of its phase sets. On a 52-sample chromosome with 25 founder states and about
945 phase sets per sample the phase task took 89 s and 0.70 CPU-hours instead of 309 minutes and 137.65 CPU-hours, and
released an identical phase product (every call, flag and phase break); with 32 states it took 183 s and 2.3 CPU-hours
(on a 76-CPU node) instead of 5 hours 40 minutes and 608.9 CPU-hours (on a 112-CPU node), again with an identical
product.

**Output format changes.** The final count step's cycle fields are removed. They have been constant since 4.12, whose
final count step only deletes founders and so cannot close a count cycle; a constant column reads as a measurement
that never fires.
- `painting/founder_counts.csv` loses the columns `count_step_cycle` and `count_step_cycle_undecided` (always false
  where a count step covered the component). The later columns move up by two; read the table by column name.
  `undecided_count_cycles` keeps its meaning and its values: it is the sum of `join_undecided_cycles` and
  `feedback_join_undecided_cycles`, the only terms that could be non-zero. The join columns are unchanged.
- `ties/summary.json` loses `ambiguity.assembly.count_cycles` and `ambiguity.assembly.undecided_count_cycles` (always
  zero). `count_calls`, `ambiguous`, `tied_refit_decisions` and `tied_refits_disagree` are unchanged.
- The stored count call of an assembly component (`CountCall`) loses `count_cycle` and `cycle_undecided`. Assembly
  checkpoints of earlier trees are not read: their records carry another code identity, so a resume is refused and the
  run is repeated in a new output root.

The count-cycle column removal leaves founder panels, counts and paintings unchanged.

Validation: the test suite (4,201 tests), the stage tier (347 tests in four shards; 2 skipped) and the documentation
check pass. The end-to-end matrix of thirteen cases (twelve simulated crosses and TroMau) was compared with the latest
earlier run of each case (4.13.0 for six, 4.11.0 for seven). Every case completes and no worker is lost.
- Seven simulations have identical truth metrics: two backcross designs, a reversed run and a sample-order
  permutation of the 80-sample cross, and three further designs.
- Four differ by a few alleles: the standard cross (founder errors 397,520 to 397,516, genotype errors 16,769 to
  16,768, phase sets 404 to 403), the stress cross (founder errors 3,928 to 3,926, missing founder alleles 3,467 to
  3,470), a three-generation cross (missing founder alleles 2,660 to 2,663) and a sibling-rich cross (correct parent
  states 96 to 97).
- The 30-founder cross is slightly worse on founders: 2,781 to 2,955 wrong alleles (2,453 to 2,589 per million called,
  with 0.7% more alleles called), genotype errors 1,300 to 1,303, phase switch errors 26 to 28, correct parent states
  23 to 24. This is the cost of discovery's screened follow-up moves.
- TroMau: the released pedigree (Tier B) is identical for all 116 samples, and one sample loses its Tier A pair.
  Founder counts are unchanged on every chromosome. Phased heterozygous calls rose by 0.03% (96,815,384 to
  96,842,678), with 27,596 phase sets instead of 28,241 and 889 crossovers instead of 892. On one 76-CPU node the run
  took 4,787 s against 4,785 s.
- Tied incoming roots: twelve components of real runs were replayed from saved inputs under the earlier and the new
  rule. Released rows, objective, release families and count decisions are identical in all twelve; the largest cost
  16,723 CPU-seconds instead of 41,705.
- Several nodes: 48- and 320-sample simulations reconstructed by several invocations on distinct nodes release the
  results of one invocation, also when one of them is killed mid-run. One chromosome of each of two real crosses (52
  and 304 samples, about 2,700 blocks each) ran on four 112-CPU nodes: every selection round of the first was complete
  after 50 minutes against 126 on one node, and discovery of the second after 23 minutes against 49. More than 30
  node-hours of concurrent appends to one container showed no fault.
- Family phase: the matrix was run before and after the release change above. Every published file of the thirteen
  cases is identical in the two runs (TroMau: 183 files, 11.8 million VCF records), and TroMau's phase stage took 406
  task-seconds instead of 983.

Limits seen in those two runs. A join batch or a refinement component is one unit, run by one invocation, and a node
claims every ready task it has CPUs for, so the last tasks of a level can sit on one node while the others wait. The
longest units are now level-1 join batches (up to 65 and 100 minutes; their search is bound by one thread, and on
saved batches it is 10 to 12% faster than under 4.13.0), where 4.11.0 had level-1 refinement components of more than
six hours (now at most 33 minutes).

## 4.13.0 (10 November 2026)

This release bounds the founder-count scaling of five searches, so that no stage grows faster than K³ except where
stated, and makes small cohorts use more of a node. 4.13 does not read or resume 4.12 checkpoints or exports.

**Local selection: the context screen.** Exact drop and add context codes come from one matching-sensitivity pass, and
all merge pairs are screened with compiled independent-minima certificates and lazy exact resolution. Each round
requests at most 128 exact codes for uncertified merge starts, shared between ordinary and K−2 consensus merges; the
remaining starts keep optimistic screen values, and admitted refits are judged by their exact objective. This declared
screening approximation removes the quintic exact-matching term but keeps a quartic worst case (certificates and start
de-duplication). On a deliberately sparse, conflicting context it selected a different panel with a 0.226-nat
objective loss and unchanged coarse truth-distance counts, so the allowance is not a quality error bound.

**Founder refinement: the count pass.** Exact per-sample score misses and admitted shadow-family score vectors are
reused through the component's shared ledger, including continuation branches. Admission is atomic and limited to
4,096 entries and 64 MiB (vectors, panel keys and bookkeeping), reserved once per component; completed-fit
publication keeps precedence, and a saturated ledger recomputes the same score. Count rules, bootstrap streams,
shadow-family admission and exchange scoring are unchanged. The pass still has no cubic worst-case bound.

**Founder refit.** Allele refit uses exact constant-time joint-gain and conflict identities on finite likelihood
tables, marks shared-row ambiguities directly, and certifies compatible tied moves over the rows eligible for tie
tests. Stable linear ordering and tie-component labelling keep the whole-round ambiguity rule. Finite work is cubic
when tied moves are certified, with a documented quartic-per-sweep exception; zero-likelihood tables keep direct
evaluation. The serial reference implementation moved to the external test suite.

**Discovery: gauge cuts.** One polynomial gauge-cut search runs at every founder count (the exhaustive small-K route
and `exact_cut_max_k` are removed). It adds 16 reproducible Goemans-Williamson starts from a Burer-Monteiro mixing
solve and maintains exact integer move gains incrementally. Oversized best-weight cut families keep a canonical,
founder-content-ordered prefix and report the truncation as a search limitation instead of being dropped. These
proposal changes can alter discovery panels; the cut search does not certify a global maximum.

**Painting: label continuity.** Tied-peer checks are charged to the same per-iteration budget as ranked candidates.
If a tie class cannot be decided within the budget, its repair is withheld, the search stops with
`check_budget_reached`, and `painting.relabel_class_budget` is recorded. Look-ahead reserves allowance for waiting
results, so an iteration makes at most 2b−1 trial decodes for budget b. Decisions within the budget are unchanged;
an over-budget class can withhold a correct repair.

**Small cohorts.** Refinement sample kernels size their team by work (component sites × K² per sample) instead of a
fixed eight samples per thread. For cohorts that fit one 64-lane block, selection groups at most four blocks per task
with two tasks per CPU and lets grouped tasks grow to the cohort's useful width, and family phase admits sample releases
as elastic children whose independent forced-cut intervals use CPUs freed by earlier samples; larger cohorts keep their
previous routes. The scheduler keeps its sorted pending queue between passes, recomputing every key. On a 48-sample,
three-contig simulation on one 76-CPU node the run took 545 s instead of 604 s with identical outputs.

Validation: the test suite (4,059 tests), the stage tier (344 tests in four shards; 2 skipped) and the documentation
check pass, and the reduced end-to-end matrix of six cases was compared with 4.12.0's. The five simulations (k30, std,
backcross, a reversed run and a sample-order permutation) have identical truth metrics. On TroMau the released pedigree
(Tier A and Tier B, 116 samples) is identical. The local panels entering assembly changed on chr3, chr9 and chr18, so
3.1%, 2.8% and 1.1% of those chromosomes' founder cells differ, mostly as founder relabelling (the same alleles at 77%,
97% and 91% of the differing sites); founder counts are unchanged on every chromosome. On chr3 the change comes from the
new gauge-cut starts alone: with 4.12's discovery code 4.13 reproduces 4.12 at every stage, and exactly one of 642
discovery blocks differs (13 founders instead of 16). Phased heterozygous calls rose by 0.11% (96,710,927 to
96,815,384), with 28,241 phase sets instead of 28,128, and 892 crossovers instead of 891. A TroMau run of the integrated
tree reproduced the matrix tree's released results; it also showed that letting one-thread maps return CPUs above their
unfinished work slowed evidence reading threefold, so that change is not included. Each design and its port was reviewed
adversarially, as was the integration.

## 4.12.0 (10 November 2026)

This release reworks the package's structure after an architecture review of the whole code base (twelve reviews by
two models), retires code kept only to reproduce 3.5's bits or to compare with it, and widens the contig lanes.
Scientific decisions are unchanged except where stated below (the shadow-family rule, which changed no released
result on the saved units compared).

4.12 does not read or resume earlier checkpoints or exports. Re-run earlier runs with 4.12 in a new output root.
Checkpoint layout markers, invocation records, export manifests and VCF/BCF headers now identify release 4.12;
mismatched or unversioned earlier artifacts fail before payload decoding. Kept-site sample VCFs now have a single
VCFv4.2 declaration, matching all other current VCF exports. The long-superseded `migrate-config` command is retired in
4.12. Earlier release notes below describe their original commands and document locations. A derivation's code identity
covers the engine and store modules its code imports, so execution changes such as the lane width also recompute earlier
checkpoints, by design.

**Wider contig lanes, largest first.** A run now gives each concurrently processed contig a lane of 16 cores (8 in
4.11), so a 76-CPU node runs four contigs at a time instead of nine, and contigs start in decreasing order of their
size estimate. On the N = 320 depth-5 simulation on single 76-CPU nodes, the whole run took 3 h 52 min with nine
lanes, 4 h 19 min with four, and 4 h 02 min and 4 h 04 min under 4.11.0; repeated runs of one configuration differed
by about 9% between nodes. Every evaluation summary was identical.

**One shadow-family rule.** Assembly join and founder refinement share the total-gated join rule. Shadow
families are enumerated only after the total test passes; passing over-budget
families keep their conventional incumbent and count_unverified. Opposed shadow
verdicts withhold without forcing bootstrap ambiguity. The refinement-only
shadow-envelope certificate is removed. Join shadow-family representative_prior
changes from `smaller_K_then_exact_minimum_switches` to `smaller_k+nearest_shadow`,
the existing refinement spelling. Other minimum-switch continuation priors retain
their names; this vocabulary does not enable reading earlier run checkpoints. The join incumbent follows
RowGeometry.reverse, matching refinement; only reversed-convention runs can
change due to this incumbent correction. Exact ties among enumerated shadows
also follow the reversal convention: both previous callers always selected the
first canonical member. Scoring order remains canonical. Policy selection used 2,193 matched
saved units with identical released results and truth scores for both policies.

**Retired code.** Final-level founder additions (off since 4.9) are removed with their setting. So are the decision
traces that every acceptance pass built and discarded, the phase and recombination trace instrumentation, the test-only
fixed-K sweep route (fitting has one route), the supplied-evidence pedigree pilot, the unused storage framework and the
summation replicas kept to reproduce 3.5's rounding. Decisions that need order-independent operands use an exactly
rounded sum; other reductions use one declared expression. Decision-level replays of the changed reductions on saved
inputs compared about 530,000 decisions and none changed. Comparisons with 3.5, validation-only reference
implementations and the workflow receipt moved to the test suite, outside the package.

**Structure.** The scheduler is split by state ownership into the scheduler, the memory governor, the CPU authority,
the contig lanes and the thread maps. Stage tasks live in their stages (`stages/<stage>/tasks.py`) instead of the
command runner, with one identity per product, and painting's checkpoint logic moved from science into its stage.
Each stage has one parameter record (`*Parameters`), without mirrors in the contracts; every parameter value is
unchanged. Science creates no thread pools of its own: it uses the engine's thread maps. Comments and parameter
provenance state current meaning, units and rationale instead of history (the generated parameter reference shows
them). Timing fields that were always zero are removed from the refinement and join records.

**Execution.** The store releases its append lock before the final fsync of a publication and groups the
acknowledgements, so concurrent publishers to one contig no longer queue behind each other's fsync. Run records give
each task's publication lock wait and fsync time, with at most one contention warning per contig. Assembly refinement
re-reads its live CPU share at every phase boundary, so cores freed by finished siblings reach long refinement tails;
run records show each refinement unit's threads over time.

**Configuration.** The AulStu example configuration (`configs/astcal.toml`) lists its 22 autosomes; it named a preset
that 4.0 removed.

VALIDATION_PENDING

## 4.11.0 (9 November 2026)

This release follows an independent review of every change since 4.4.0 that had not been reviewed (4.4.0's late fixes
through 4.9.0, and the items 4.10 deferred). The review found no further scientific defect in the supported paths
(the silent painting fault of 4.7.1 to 4.9.0 was found by 4.10's release matrix and is fixed in 4.10.0); the fixes
below are operational, and 38 of the 39 release-matrix cases are identical to 4.9.0's in every scientific, numerical
and diagnostic value; their output directories also hold the new lock files (the 39th case was still running at
release).

**Prepared and observed-count decodes overlap.** Checkpoint decompression releases the GIL under one process-wide
setting made once, instead of saving and restoring it around every decode under a lock, and the pedigree screen's
prepared records and the decision's observed-count ranges are read with thread-shared stores. On the N = 320
depth-0.5 checkpoints on one 112-CPU node (warm file cache): the decision's observed totals 36 s -> 9 s, the
screen's 22 prepared records 7 s -> 3 s, identical values. (4.10 read them on separate threads, but their decodes
still ran one at a time.) Items of a one-thread map no longer borrow a worker lane's idle CPUs for their decodes:
their siblings run beside them, so each would have multiplied the lane's threads.

**Export keeps a consistent pair on failure.** Replacing an indexed VCF builds the new index first, then, under a
lock on the file, keeps the accepted data under a link, moves an index of the other kind aside and renames the new
data and index into place. A failure before the new index is in place restores the accepted data and indexes; after
it, the new pair stands and no index of the other kind is left beside it (4.10 removed the previous indexes before
moving the data, so a failed rename left the accepted file without its index). This holds for handled interruptions,
not for a hard kill. Writers of the same file serialise these moves, so a failing duplicate writer never restores
older data beside another writer's new index. The lock is a hidden, empty `.<file>.lock` beside each indexed output
file: persistent coordination metadata, not a product, absent from the export manifest but present in a raw listing
of the directory. An export destination holding only such locks counts as empty. When a change to the run's export
options replaces a contig's files, the contig's previous export record no longer certifies them, so an interrupted
export is rendered again when the run resumes.

**Bounded painting memory.** Label continuity builds the emissions of uncached sample batches one batch at a time as
they are consumed, instead of all at once.

**Observation counts.** The whole-window counting path is taken only for observation masks (allele-count arrays keep
the per-row path, which 4.10's fast path did not accept), and its window is sized by the selected sample rows with no
1,024-site minimum, so a gather holds at most 2^24 cells (one site column when the selected rows alone exceed that).

**Clean interruption.** Worker processes give Numba's backend initialisation an unnamed lock, so a terminated run no
longer leaves that lock's semaphore for the resource tracker to report. A reply segment whose close was deferred
(4.10) is retried after each later reply and when the worker goes idle.

Validation: the test suite (3,810 tests, on RHEL 8 and on RHEL 9), the stage tier (341 tests), the workflow receipt
and the documentation check pass. Reviewed adversarially: four review workstreams and an integration review.

## 4.10.0 (9 November 2026)

This release removes serial stretches from the whole-genome stages (pedigree to the end) and from a simulated run's
initialisation. On the N = 320 depth-0.5 simulation, one 112-CPU node after the 22 chromosome tasks, the genome-wide
run took 2,059 s at a mean of 26 busy CPUs with 4.9.0 and 970 s at 53 with this release; the whole run, chromosome
tasks included, took 9.32 node-hours with 4.9.0 and 7.04 with this release (the chromosome stages' code is unchanged:
part of that difference is the nodes they ran on). The changes schedule, parallelise or stream work, and one fixes a
fault that could silently corrupt a run's painting in releases 4.7.1 to 4.9.0 (next paragraph); the release
matrix's published products are identical to 4.9.0's (below), and the replays named below are byte-identical.

**Fixed: corrupted painting in some runs of 4.7.1 to 4.9.0.** Numba names the native loop of a parallel kernel after
the address of its intermediate representation and its argument types. Worker processes started alike allocate alike,
so two different kernels with the same argument types, compiled in two workers, could receive the same name, and a
process that loaded both from the run's kernel cache then ran one kernel's loop for the other. The pair that met this
in practice is painting's forward and bin-reversed structured emissions (both since 4.7.1). When the variant a run
called collided, the decode's posteriors collapsed in almost every sample batch while the fitted model stayed
correct, and the released painting, the pedigree evidence, founder refinement and phase of that run were degraded. It
happened at random, per run with a newly compiled kernel cache: in about one run in four of one test design, with
4.9.0 and with this release's first candidate alike, and in three of the 39 cases of that candidate's release matrix.
Each process now gives its parallel loops names of its own; no computed value depends on a name. With Numba 0.68, an
affected run showed low-posterior abstentions (`LOW_POSTERIOR_ABSTENTION` in `painting/sample_segments.csv`) over
most of its painted sequence (58-73% in the three affected matrix runs, at most 5% in 39 unaffected ones, 1.3% on the
TroMau cross), with several times as many segments. That is a warning sign, not a complete test: a collision
affecting only some batches or contigs can stay below it, a run reusing an affected kernel cache repeats the fault,
and weak evidence alone raises abstention. Analyses run with 4.7.1 to 4.9.0 should be rerun with this release,
keeping the old outputs for comparison; deleting a kernel cache does not repair published results (this release's
caches are separate from older ones).

**Family phase uses the node.** A component's two label models are solved at once, each on a share of the task's
CPUs, and each model's coordinate ascent updates samples in conflict-aware waves: a sample's update writes its own and
its affected descendants' rows and reads those rows and their linked parents, and samples whose sets do not meet an
earlier one's in the sweep order run together, giving the sequential sweep's result. The uncertainty, the release
(site windows of 65,536 sites, copied once per solve), the agreement and the phase-set starts are mapped over the
task's CPUs. Chromosome 3 alone on a node (saved 4.9.0 inputs): 645 s -> 117 s, the same phase product.

**Export.** The sample-phase VCF text is formatted natively instead of one Python string per cell, and VCF text goes
straight to htslib's BGZF writer with the blocks its record writer makes (a phase set beyond int32 is written missing,
as htslib does); a writer with a share of T CPUs compresses and indexes on T - 1 htslib workers beside its own thread.
Chromosome 3's export (saved 4.9.0 inputs): 120 s -> 30 s, files and indexes byte-identical.

**Pedigree and relatedness.** The screen builds its certified selective bounds with array operations (the same
bounds, bit for bit). Relatedness counts observed sites one site window of all samples at a time (its genome-wide
task: 45 s -> 24 s in the run above). The screen now reads its contigs' prepared records, and the decision each
contig's observed totals in sample ranges, on separate threads; their large decodes still run one at a time, so
neither is faster yet (the decision's step was slower than when contigs were read whole; results are identical).

**Resume.** A completed contig's discovery, selection and assembly are recognised from their published records instead
of being expanded again; code identities are cached per process; binding and completion checks read their records
concurrently. Creating a checkpoint directory under concurrent claims checks its status again after FileExistsError
(a bounded retry; an entry that is not a directory still fails).

**Initialisation.** A simulation's read blocks write into one shared array per contig, owned by its task like other
shared values, instead of returning their counts through the coordinator; the pedigree rounds and the reads share one
worker group; templates and founders are read and drawn concurrently; completion and resume check the reads' records
without decoding them. N = 320: 73 s -> 44 s, with identical simulated data.

**Operational.** A worker that cannot close a reply segment because a view of it is still alive collects and retries,
then defers the close, instead of failing the run (one release-matrix case failed so in assembly; the code was
unchanged since 4.9.0); the shared-memory copy kernel declares its read-only source.

Validation: the test suite (3,808 tests), the stage tier (341 tests), the workflow receipt and the documentation check
pass. All 39 cases of the release matrix are identical to 4.9.0's in every scientific, numerical and diagnostic value:
36 from the first candidate and, for the three that candidate's painting corrupted (above), reruns of this release.

## 4.9.0 (8 November 2026)

This release removes the remaining long serial stretches of a single chromosome's run. Chromosome 1 of the N = 320
depth-0.5 simulation on one 112-CPU node (`run --stop-after painting`, then `reconstruct`): 1,094 s at a mean of 70
busy CPUs (4.8.0) -> 875 s at 74. The final level no longer tries founder additions (below); every other change
schedules or parallelises work, and the released painting and founder refit of that chromosome are identical to
4.8.0.

**No founder additions at the final level.** Founder refinement's count pass at the final level tried, after its
deletion rounds, adding a founder (each row duplicated and refitted, then tested by the count rule). Across 42
validation runs (418 chromosomes: simulated designs including K = 30, backcross, F3, sib-rich and family designs,
TroMau and the controls) no such addition was ever accepted (0 of 402 decisions; 238 ran full refits), while an
addition round took about two thirds of a whole-chromosome component's refinement: chromosome 1's final level
251 s -> 80 s with the same panel. The new internal refinement parameter `final_additions` (off) gates them.
Founder panels are unchanged wherever additions would have been rejected; the refinement records no longer list
addition decisions, and the ambiguity of an addition call (10 of the 402) is no longer reported.

**Discovery blocks grow into freed CPUs.** A discovery block reads its task's live CPU share at every kernel choice;
it fixed its thread count when it started, so the heaviest blocks, started beside hundreds of siblings, ran
serially to the end (chromosome 1 block 287: about 50 s; 11 s on 32 threads, identical results).

**Selection's lone blocks grow again.** Every block alone in its task may grow into CPUs its siblings leave, as in
4.7 (4.8 kept those the cost model rated light on one CPU, and mispredicted ones ended refit rounds alone).

**Serial steps between stages.** Assembly builds its evidence views 32 samples per gather and fill on the task's
CPUs plus its lane's idle ones (one sample per kernel call ran serially at every chain step and on resume), and a
large compression or decompression call borrows the lane's idle CPUs (a single-CPU report or refit task decoding a
large record). Resuming a completed chromosome for founder refit: 51 s -> 44 s.

Released records of chromosome 1 on identical simulated data are identical to 4.8.0's (painting, founder refit and
its summary). The test suite (3,805 tests), the stage tier (341 tests), the workflow receipt and the documentation
check pass.

## 4.8.0 (8 November 2026)

This release removes serial stretches from a single chromosome's run so that one chromosome uses the whole node.
Chromosome 1 of the N = 320 depth-0.5 simulation on one 112-CPU node (`run --stop-after painting`, then
`reconstruct`): 1,409 s at a mean of 55 busy CPUs -> 1,094 s at 70. Simulated reads change for a seed (below); every
other result is scientifically identical to 4.7.1: the other changes reorder, batch or parallelise work.

**Simulated reads in parallel.** Each sample draws its reads from its own stream (`simulation.sample_reads`: a seed
drawn on the contig's read stream, and the sample's index); the factors shared by all samples still come from the
contig's stream. Both samplers (independent sites and paired-end fragments) therefore simulate blocks of samples on
all CPUs with the same counts at any worker count, from read-only inputs shared once; a gather writes the contig with
the run's CPUs compressing. Meiosis tasks carry only the parents their own children use (4.7.1 sent each task the
whole parent cohort: about 4.4 GB pickled by the coordinator for chromosome 1). The read laws, pedigree, meiosis and
truth are unchanged, but a seed's read counts differ from 4.7. Simulated input for chromosome 1: 33 s -> 7 s.

**Kernel compilation per overload.** On a cold kernel cache each overload compiles as its own task, and the cache lock
covers one overload's compilation (and every index save) rather than a whole function: a function's overloads, and
the stages waiting for them, no longer queue behind one another. Every declared kernel cold: 204 s -> 88 s.

**Evidence encoding in parallel.** The coded genotype-likelihood encoder plans and fills samples on the evidence
task's CPUs and checks values in larger windows (bit-identical output): 15 s -> 1 s for chromosome 1.

**Selection rounds end together.** A round's blocks are grouped into at least eight tasks per CPU, chunks shrink as
the round's remaining work shrinks, and each task declares its cost so the heaviest are dispatched first (they were
dispatched in block order). A light block alone keeps one CPU.

**Checkpoint reads and writes.** A compression or decompression call takes at most one thread per MiB it handles:
stores opened with a whole node's threads started 112 Blosc threads for every small record (a selection round's
1,647 panel reads: 14.2 s -> 0.8 s). A record read checks its envelope on the same decode that selects its fields,
and loops over a stage's block records read through one index snapshot and one file handle.

**Fewer serial steps between stages.** A task's large result is copied into shared memory with its CPUs plus the
lane's idle ones; the assembly depth keys are gathered 32 samples at a time and hashed in parallel; code identities
read the package sources once, concurrently, also before a runtime starts.

**Fix.** The command line's isolated start-up process now imports from its parent's module path. Started in a source
checkout's root, it imported that checkout's package rather than the one running the command (the simulation identity
check refused the result).

Released records of two 4.8 runs of chromosome 1 on identical simulated data, before and after the read and
scheduling changes, are identical (painting, founder refit, evidence; the final assembly differs only in timing
fields). The test suite (3,805 tests), the stage tier (341 tests), the workflow receipt and the documentation check
pass.

## 4.7.1 (8 November 2026)

This release removes serial work from painting and gathers in-memory evidence in parallel. Every result, report and
checkpoint value is scientifically identical to 4.7.0: the changes copy, cache, fuse or reorder work only.

**Painting builds from memory.** Each component's evidence at its panel sites (every sample's genotype-likelihood
codes and its likelihood dictionary, far smaller than the emissions) is read once into memory, charged to the 8 GiB
emission budget whenever the copying-model estimate and the decode keep the packed emissions they would keep without
it. The estimate's updates, label continuity's batches and the decode then build emissions with a parallel in-memory
gather and emission kernel instead of serial reads of the stored evidence.

**Label continuity without serial rebuilds.** Label continuity keeps each sample batch's part emissions for its whole
search when they fit its budget. In 4.7.0 a component whose emissions did not all fit rebuilt every batch from the
stored evidence for every candidate it checked, one sample at a time on one thread. Now:
- the batches that fit are kept (not all or none);
- without the component copy, the search reads its own copy of the selected evidence under its budget, for every
  sample or for the leading batches that fit;
- one rebuild serves several candidates: after a candidate fails, the next ones in ranked order (1, 2, 4, ... per
  iteration, within the check budget) are decoded from the same emissions, and count as checks only when the search
  reaches them;
- several candidates of a batch are decoded in one call, as many as the budget the kept emissions leave can hold, so
  each parallel call has more samples than threads;
- when every batch is kept, they share one array and each decoding call takes as many samples as that budget holds
  (all of them on the test below), not one batch.

**Fewer serial steps between kernels.** One kernel tempers each emission window (informative sites, the design
effect of each whole bin, the division) and writes the batch's arrays in place; the mirrored Viterbi convention reads
the reversed emissions in place instead of copying them; path labels and switch counts are kernels. A gather of
in-memory coded genotype likelihoods or observation flags of at least 16,384 sample-site cells runs as one
sample-parallel kernel, in every stage, and a selection of stored dictionary rows spanning no more rows than it
requests is read in one range read.

**Task results copied in parallel.** A pool worker copies a task's large result arrays into shared memory before it
releases the task's CPUs, with the task's whole team, instead of on one thread afterwards; a large assembly result
held a whole node on one CPU while it was copied.

Painting chromosome 12 of the N = 320 depth-0.1 simulation on a whole 76-CPU node, kernels compiled beforehand (for
4.7.0's second case compilation is under 3% of the time), with an identical painting in every case:
- the emissions kept (the usual case): 88 s at 34 busy CPUs on average in 4.7.0, 54 s at 58 now;
- label continuity keeping no batch: 647 s at 7 busy CPUs in 4.7.0, 69 s at 55 now;
- the same chromosome at depth 1: 29 s at 21 busy CPUs in 4.7.0, 11 s at 55 now (the same total CPU);
- chromosome 7 of the real TroMau cross: 6.3 s in 4.7.0, 3.3 s now.

The test suite (3,806 tests), the stage tier and the workflow receipt pass.

## 4.7.0 (8 November 2026)

This release shortens the run's critical path and uses every allocated CPU. Founder calls, counts, panels,
paintings, pedigree calls and phase are scientifically identical to 4.6.0. These reports change:
- count-step decision records of rounds skipped by a bound carry the proven bound instead of a computed margin and
  band, and those rounds publish no `count_unverified` record (about half of them disappear);
- `pedigree/search_diagnostics.csv` has a new column, `m2_panel_capped`, and children whose parent pairs are capped
  have shorter explanation and candidate tables;
- `painting/painting_model.csv` has a new column, `ContinuityBudgetReached`.

A new output root is required: kernels and painting and pedigree parameters were added, so checkpoint identities
differ from 4.6's.

The supported mean read depth is now stated: 1x per sample or more, with releases also validated at 0.5x
(see the README).

**Every allocated CPU.** No stage limits itself to a fixed thread count any more. Discovery block tasks and the
evidence, relatedness-family and relatedness-pair tasks accepted at most 16 CPUs, and shared-array copies used at
most 16 threads; they now take the whole allowance the scheduler grants.

**Relatedness: the family fit grows its thread team.** The family partition is fitted once for the whole genome,
after every contig's evidence and before discovery finishes. It started with the CPUs free at that moment, often
one, and kept them: in the 4.6 N = 320 depth-1 run it held the critical path for about an hour on one of 112 CPUs.
It now re-reads its share at every agglomeration batch, move sweep and support unit, as other stages do. The family
product is identical; the fit took 478 s on one thread, 49 s on 16 and 24 s on 112.

**Every kernel releases the interpreter lock.** Several stages run kernels from concurrent threads of one
process: the pedigree screen's chromosome tasks, assembly's component team, concurrent refits and join passes, and
the evidence site-quality and read-model pools. A kernel that held Python's interpreter lock ran alone, however
many threads were waiting. All 653 kernels are now declared `nogil` (260 already were). Kernels allocate their
scratch per call and concurrent threads write disjoint outputs, so results are unchanged. Replays of the 4.6
N = 320 depth-1 run: chr22 and chr10 assembly (all cores) take 947 s and 866 s instead of 1,475 s and 1,353 s,
with the same decisions; painting and evidence preparation are identical.

**Pedigree: the slowest chromosome uses the CPUs its peers free.** The selective screen runs one task per
chromosome in each query phase, and each phase waits for the slowest. A task's parallel kernel kept the thread team
it started with, about a twentieth of the CPUs, for the whole call; the largest chromosome then ran alone on that
share while the rest of the node idled. Each query now runs in 32 chunks of parent pairs and re-reads its share
before each chunk. Pairs are scored independently, so the screen product is identical. Replaying the 4.6 depth-1
screen on 76 CPUs: 1,126 s with 23 CPUs busy on average, now 521 s with 53 (the interpreter-lock change alone gave
1,063 s).

**Founder refinement: a summed count-deletion bound.** A count-deletion round is skipped when an optimistic bound
proves, for every candidate, that the larger panel is kept and that the call is not ambiguous. 4.6 bounded each
sample separately, and each sample could drop a different founder at every block. That bound almost never fired
(0 of 1,385 recorded decisions). 4.7 sums the bound over samples at one dropped row per block, the same for all
samples, as any panel with one founder fewer must. It covers the point margin, the leave-one-sample-out margin and
every bootstrap replicate of the decision's own stream. In replays of the 4.6 N = 320 depth-1 run:
- chr22: 61 of 239 deletion rounds are skipped; refinement windows fall 22-34% by level (31% overall);
- chr10: windows fall 12% at the first two levels.

The decisions, founder rows and counts are unchanged. A skipped round runs no shadow-family evaluation. So a round whose
shadow family would have overflowed its evaluation budget is no longer reported `count_unverified`: the bound proves the
decision for every shadow. In the release matrix 318 such records fall to 152. If a best-F restore needs the adjacent
smaller fit, the skipped round's refits run first, under the same label.

**Painting: emissions built once.** The tempered emissions do not depend on the copying model's parameters, so its
8 EM updates now read one packed copy instead of rebuilding them each update. Label continuity likewise keeps each
sample batch's part emissions for all its candidate checks instead of recomputing them per candidate. Both caches
share a budget of up to 8 GiB per contig; they are not reservations. The painting product is identical. Painting
times (wall / CPU):
- chr22, depth 1: 183 s / 1,509 CPU-s to 43 s / 712 CPU-s;
- chr9, depth 0.25: 721 s / 4,904 CPU-s to 103 s / 2,214 CPU-s;
- chr12, depth 0.1: 4,824 s to 450 s wall.

**Painting: interrupted work is kept.** Painting publishes progress records as it goes: each component's copying
model, its label-continuity outcome and its decoded samples in fixed blocks of 16. A resumed run restores them and
decodes only the missing samples, on any number of CPUs. The records do not depend on the CPU count, so a run and its
shell workflow still publish the same records. The product is identical to an uninterrupted run.

**Painting: fewer logarithms and cheaper evidence reads.** Each bin's emission now takes one logarithm of the
product of its sites' clamped probabilities (rescaled before it can underflow) instead of one per site; the
emissions move in the last digits (at most 4e-14 on chr22), and every released painting value is unchanged. Reading
a sample's coded evidence rows marks the requested codes instead of sorting them; the rows are identical. Painting
chr22 at depth 1 on 8 CPUs: 92 s / 530 CPU-s before, 64 s / 403 CPU-s after.

**Painting: label continuity is bounded on long chromosomes at low depth.** Label continuity repairs label
exchanges between founder rows by trying relabellings and re-decoding every sample under each. Its last iteration
must show that no remaining candidate is acceptable, so it re-decoded every candidate that passed the screen: on long
chromosomes below 0.5x that is hundreds of whole-chromosome decodes (621 checks for 3 accepted moves on chr12 at 0.1x),
and painting time grew with the square of chromosome length. An iteration now re-decodes at most
`label_continuity_check_budget` (64) candidates in ranked order; reaching it ends the search, and the component is
flagged (painting model table, ContinuityBudgetReached). Candidates ranked below the budget are not checked. At the
release-matrix depths and N = 320 at 0.5x and 1x, no component checked more than 8 candidates, so their paintings are
unchanged. On chr12 at 0.1x (76-CPU node) the search makes the same 3 moves and the painting is identical, with
84 checks instead of 621: painting takes 144 s instead of 470 s (4.6: 4,763 s); chr9 at 0.25x is unchanged (the
budget is not reached).

**Family phase: cheaper allele-evidence terms.** The per-site evidence terms select precomputed genotype-column
steps directly instead of gathering columns. The results are bit-identical. Phase CPU on chr22 falls 17%
(681 to 565 CPU-seconds).

**VCF input: one pass per window.** Each window of a contig is read once: records are admitted, their allele depths
decoded and duplicate positions dropped in the same pass (4.6 scanned every window twice). The decoded reads and
the payload identity are identical; a TroMau contig decodes in 4.0 s instead of 6.3 s on one worker.

**Pedigree: a per-child cap on scored parent pairs.** At very low depth the admitted M2 parent pairs grow with the
square of the eligible parents (about 50,000 per child at 0.05x and N = 320), and scoring them became cubic in N.
A child now scores at most `max_pairs_per_child` (256) pairs: the best by bound gap, with whole tie classes, and
pairs with an unscored gap always kept. When the cap binds, the child is flagged (`m2_panel_capped`). In the release
matrix it binds in 8 of 39 cases (4 children of 112 in f3, 19 of 116 in each of the seven actm3 variants): their
pedigree calls are unchanged, and their explanation and candidate tables list the 256 pairs scored.

Measured and not adopted:
- a second search direction only for promising refits (it changes the winning refit in 17-31% of large-component
  rounds);
- screening acceptance candidates (no exact bound cheaper than the exact gain);
- sharing proposal templates across refits (their inputs never coincide);
- fused multiply-add (about 1% faster);
- a fused site-quality objective (no faster);
- discovery start races, at birth groups or at every fit call (they change the discovered panels: at depth 0.5,
  56 of 100 blocks changed and the fit was worse in 33 of them, better in 16);
- a local-selection trial race (identical calls, but slower at depths 1 and 5, faster only at 0.5 and below);
- incremental merge scoring in relatedness (the thread-team growth above removed its cost);
- stopping label-continuity checks at a certain rejection (identical decisions, 11% less CPU but 21% more wall time
  on chr12 at 0.1x: rejected candidates fail late, and smaller decode calls parallelise worse).

**Validation.** The 39-case release matrix and whole-genome N = 320 simulations at depths 1 and 0.5 were rerun and
compared with 4.6 cell by cell:
- every case completes; founder accuracy against truth is identical wherever it is scored;
- 31 of 39 matrix cases, all seven TroMau cases among them, and both N = 320 runs have no scientific difference,
  only the report changes above and continuous values within 1e-12 relative (painting emissions computed with one
  logarithm per bin reach phase and pedigree);
- the other 8 cases differ only where the pair cap binds, in pedigree explanation and candidate tables; their
  released pedigree calls are identical;
- N = 320: founders 5 and 79 alleles per million wrong at depths 1 and 0.5, every parent state and all 600 pedigree
  edges right, phase switch errors 31 and 32, as with 4.6.

The test suite (3,797 tests) and the workflow receipt pass; the stage tier (301) passed on the candidate before the
check budget was added. Wall time: the N = 320 critical path falls from 3.7 to 2.3 hours at depth 1 (relatedness from an
hour to 6 minutes; pedigree, phase and export faster) and from 3.5 to 2.9 hours at depth 0.5; matrix cases on the same
hardware take 16% less time (median), with real-data cases unchanged. On identical hardware a run uses the same total
CPU as with 4.6.

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
