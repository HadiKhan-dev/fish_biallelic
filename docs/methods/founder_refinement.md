# Founder refinement

Founder refinement (`assembly/founder_refinement/`) runs after every executed level of the final
[assembly](assembly.md), never in the context assemblies of local selection. It re-chooses each
phase component's founders as paths of the original prepared local rows (one row per prepared
block; the rows' alleles and evidence masks stay fixed), fitting them to a certified local optimum
of a full-site cohort objective F, and it decides the component's founder count K between converged
fits with the pipeline's [founder-count rule](assembly.md#founder-count-rule). Every level may
delete founders; only the last level (the final count step) may also add them. The refined
components feed the next hierarchy level, and after the last level they are the assembly product.
No target founder count, pedigree, cohort label or simulation truth is supplied.

## Inputs and outputs

The assembly level loop calls `refine_level(LevelInput, config, budget, records)` with:

- the contig's prepared panels and the level's join output (components with row paths over them);
- the chromosome's float32 evidence (the joins' copy of the founder view: normalised genotype
  likelihoods at observed discovery sites, exactly 1/3 elsewhere); a component reads the columns of
  its prepared panels;
- from level 2 on, the L1 context: the refined level-1 components followed by the unrefined
  level-1 join components; inside every L1 span of a component, their distinct row paths (refined
  first) offer the coarse moves;
- the cost scale (`AssemblyConfig.cost_scale = 0.5`), the count rule's settings and whether the
  level is the last one (additions allowed).

It returns the components with the same boundaries and order, their rows replaced by the refined
row matrices (K may change) and a count call set on every component, plus one refinement record per
component and the level summary. Its records live under `assembly/l<k>/refine/`
([records and census](#records-and-census)); the assembly stage then records the refined level as
`assembly/l<k>/refined` and, after the last level, `assembly/final`. Readers in
`assembly/founder_refinement/records.py`: `founder_panels(file, level)` (refined level k, or the
last refined level), `refinements(file, level)` and `refined_levels(file)`. There are no user
options: the constants below are internal (the [configuration reference](../configuration.md) lists
them), and `[execution] cores` and `[execution] memory_limit` change timing only: the window-frontier
cap is a fixed constant ([window programmes](#window-programmes)), never the memory that happens to be free.

## Objective

For a panel X of K founder paths over a component's prepared blocks, F(X) = sum_s F_s(X), where
F_s is the full-site Viterbi score of sample s in an unordered-diplotype Potts chain
(`assembly/founder_refinement/objective.py`):

- the states are the Q = K(K+1)/2 unordered founder pairs, and a state's dosage at a site is the sum
  of the two founders' pre-fill calls;
- a site emits max(log(0.99 p_d + 0.01/3), -2) - log(1/3) for the dosage d of the state, with p the
  normalised genotype likelihoods of the sample;
- every change of state costs `assembly.founder_refinement.objective.SWITCH_PENALTY = 20.0` at
  every level (growing it with the component's length could prefer a poorer founder phase merely
  because it saves sample switches); the start is free and there is no length scaling;
- only complete informative sites count: discovery sites where every local row of the prepared
  block is called, and evidence that is not exactly uniform. The mask is fixed across the candidate
  rows, so a path cannot raise its score by hiding difficult sites.

Symmetric genotype emissions and a uniform switch penalty make the reduction to unordered pairs
exact for this model. It is an internal assembly objective, not the downstream homologue painter, a
phase posterior, a genealogy model or a recombination-map estimate. Count decisions charge each
founder the cost c = s x (mean prepared-panel sites / 200) x S x B of the count rule, for B prepared
blocks and the S informative samples of the component: samples with evidence at one of its complete sites.
Every other sample (samples excluded from founder construction, whose evidence is uniform, or without a read at
a complete site) scores exactly 0 for every panel, so it enters neither the cost nor the bootstrap population.
The join's S is the batch's informative samples, the rows its evidence holds.

## Converged fixed-count fits

A fixed-K fit C(X) (`assembly/founder_refinement/search.py`) repeats Gauss-Seidel sweeps until a sweep accepts nothing. Each
sweep:

1. computes the exact normalised forward and backward block messages of the panel;
2. computes window proposals from the panel at the start of the sweep: for every founder and every
   fine window of `assembly.founder_refinement.proposals.FINE_WINDOW = 8` prepared blocks (every
   `assembly.founder_refinement.proposals.FINE_STRIDE = 4` blocks, plus the window ending at the
   last block), the canonical exact maximiser of a binned surrogate with the other founders fixed
   and the rest of the component entering through the exact flank messages; with an L1 context, the
   same over coarse windows of `assembly.founder_refinement.proposals.COARSE_WINDOW = 4` L1 spans
   (every `assembly.founder_refinement.proposals.COARSE_STRIDE = 2` spans), whose alphabet on each
   span is the distinct row tuples the founders use there followed by the refined and unrefined L1
   paths;
3. runs one exact acceptance pass over the junctions in block order, its direction alternating
   between sweeps (`assembly/founder_refinement/acceptance.py`). At every junction the best founder-pair suffix exchange is
   applied while its exact gain exceeds `assembly.founder_refinement.acceptance.TOL = 1e-4`; then
   the proposals anchored at the junction (one founder's rows on the blocks where a proposal differs
   from the current panel) are re-scored exactly against the current panel and applied best first
   (earliest on ties) while their gain exceeds the same tolerance; coarse proposals must also pass
   the [guard](#coarse-move-guard);
4. recomputes F, which must equal the start of the sweep plus the accepted gains within
   1e-9 max(1, |F|) + 1e-6 per accepted move; a larger deviation stops the run with an error.

A sweep that accepts nothing certifies the fit: every proposal of that sweep was computed against
the final panel and every move of the move set was scored exactly against it, so the panel is a
local optimum of F under this move set. F is non-convex and no global optimum is claimed. The window
proposals optimise a binned surrogate, so the certificate is relative to them; acceptance always
scores individual sites. A search that reaches `AssemblyConfig.refinement.sweep_limit = 100` sweeps
is reported (warned, and recorded as not converged), never silently accepted.

**Best of two.** Every fit runs this search twice from the same start, concurrently: once with its
first pass forward and once reversed (both then alternate; they share the first sweep's messages
and proposals). Both are deterministic local searches. The existing F tie classes and count
comparison criteria identify the surviving candidates; among remaining same-K ties the
representative uses the exact minimum sample state-change count. Both results are recorded.
This applies to every start fit and every count refit.

**Surrogate bins.** The window surrogate cuts each prepared block's sites into bins of
max(b_min, ceil(L_c / 2000)) sites for a component of L_c sites
(`AssemblyConfig.refinement.proposal_max_bins = 2000`) and sums over a bin's discovery sites the
robust emission of the join's binned model: hard dosages where every row of the block is called at
its discovery sites, otherwise the predictive pair classes with unknown alleles marginalised (so the
surrogate also reads sites that F leaves out). At levels 3 and above, b_min is
`AssemblyConfig.refinement.proposal_min_sites_per_bin = 200`; at levels 1 and 2, b_min is
max(200, ceil(chromosome sites / 2000)), shared by all the components of the level, so the many
small early components do not each receive a chromosome's bin budget.

### Window programmes

A window problem maximises the surrogate over one founder's rows on the window's units (prepared
blocks or L1 spans). The solver (`assembly/founder_refinement/proposals.py`) first tries the box bound (every sample choosing the
founder's later rows freely), which certifies windows whose incumbent is already optimal. Otherwise
an exact forward frontier programme runs: a partial path is dropped only when another is at least as
good for every completion (closed-message dominance) or when a valid upper bound on its completions
lies below a feasible value. The answer is canonical, a pure function of the problem: among paths
within 1e-7 + 1e-9 |H| of the optimum H, the incumbent if it is one, else the lexicographically
first path (units left to right, each in the order incumbent row, then ascending row index). A
programme whose frontier passes `assembly.founder_refinement.proposals.EXPLORE_FRONTIER = 256`
partial paths restarts with tighter pruning (Lagrangian suffix bounds and the best feasible value of
the incumbent, a 16-wide beam and the dual's decoded paths); pruning keeps every path within the
margin of a feasible value, so the optimum and the canonical answer are unchanged.

The frontier is capped by a fixed byte budget, `AssemblyConfig.refinement.frontier_bytes = 137438953472`
(128 GiB), the same on every machine: a kept partial path holds 8 S (Q + K + bins + 2) bytes and a programme holds
up to four times its frontier. A programme that would exceed the cap stops and keeps the incumbent for that window;
such a fallback is counted per fit and per level, warned and shown in the level's census line. The memory of a
programme is admitted in steps: it is first solved within a budget covered by its window solve's own estimate (just
past the restart trigger, at least `assembly.founder_refinement.proposals.PROGRAMME_FIRST_BYTES = 268435456`
bytes), and a programme whose frontier passes it is solved again within budgets
`assembly.founder_refinement.proposals.PROGRAMME_GROWTH = 8` times larger, each reserved from the run's memory
governor, up to the cap. A solve that stays within a budget computes exactly what a solve at the cap computes, so
results depend neither on the budgets nor on pacing.

### Coarse-move guard

A long founder edit can raise F by letting several samples drop the switch of a crossover they
share (descendants of one meiosis carry the same breakpoint), turning the founder into an
artificial recombinant. Coarse moves (L1-span proposals) therefore pass the shared-junction guard
(`assembly/founder_refinement/guard.py`): a coarse move that lowers the genotype fit E = F + 20 x switches (switches of the
samples' maximum-score paths with the fewest switches) is refused unless its F gain exceeds the
acceptance tolerance once every saved switch at a shared one-way junction is uncredited,
dF - 20 N_shared > TOL. N_shared sums over samples the smaller of the switches the sample saves and
its dropped switch events at shared one-way junctions. A dropped event is a founder transition
u -> v of one homologue that the move eliminates; it is one-way unless it leaves the edited founder
and the same path later rejoins it (an excursion, the signature of a row that is wrong there), and
shared when another sample's one-way dropped event is the same transition with an overlapping
evidence interval (the stretch between the sample's nearest informative sites where u and v
differ). Moves that do not lower E, and moves whose gain comes from unshared junctions or
excursions, pass. The guard applies only where coarse moves exist (levels 2 and above). It can
withhold a true move; it is not a confidence threshold or a proof of biological correctness.

## Founder-count decisions

Each component's count pass (`assembly/founder_refinement/count.py`) starts from the converged fit of its incoming panel and
compares converged fits one founder apart. Components with one block or one founder pass through
unchanged with the reason `single_block_or_founder`.

**Deletion rounds** (every level). The candidates are the founders without a private cell (no row at
any block that no other founder uses) when there are any, otherwise every founder. With more than
r = `AssemblyConfig.count_rule.ranked_refits = 9` candidates, every candidate's deletion is scored
exactly without refitting and the r deletions with the smallest point margin (the smaller of m and
m' on the total gain) are refitted; otherwise every candidate is. The refits converge concurrently,
and the best-F refit (exact ties: the lower founder) is decided against the incumbent by the count
rule. Its sequence part uses the shadow against the incumbent without the decided founder, so only
that founder's private cells change; the shadow is scored exactly (refinement takes no near-identity
shortcut, because its last level is final and no later level could re-split a merged pair). While
some founder has no private cell, q = 0 and every round deletes one of those founders. A rejection
ends the deletions, and panels of two founders are not reduced. A kept round also records the other founders
whose sequence part fails the test (diagnostic only: only the decided founder is tested).

**Deletion bound.** When every founder has a private cell, per-sample upper bounds U_s on F_s of any
panel with at most K - 1 distinct rows per block may decide the round without refits (each block
chooses its own rows and every diplotype restarts freely at each block; a block with exactly K local
rows takes its best one-row deletion, every other block offers all its rows): with l_s = F_s(X) - U_s, the round keeps the incumbent when sum l / c - 1/2 and
(sum l - max l) / c - (S - 1)/(2S) both exceed a small tolerance. These bounds also bound every
shadow panel, so they prove both tests for every candidate.
A round is skipped only when the bound also proves the bootstrap band: every replicate of the decision's own
bootstrap (same seed, same resampled samples) has min(m_d, m_q) >= (sum of its resampled l) / c - 1/2, so the
band's lower end is at least that bound's percentile; when the percentile exceeds the tolerance the call is proven
not ambiguous for every candidate, otherwise the round runs. The record keeps the percentile (`band_bound`).

**Addition rounds** (last level only). A round is skipped when an optimistic bound on any panel with
one founder more proves that no addition passes: either the total optimistic gain stays below the
smallest threshold c/(2S), or no sample's optimistic gain reaches c/(2S), which keeps the point
margin and every bootstrap replicate at or below zero (rejected and not ambiguous). Otherwise the
seeds are the panel plus a duplicate of founder f, for every f; with more than r founders, the r
seeds whose copy gains most from its exact best responses on all fine windows (one copy-only
acceptance pass) are refitted. The best-F refit is decided against the incumbent; its shadow
replaces every cell of the refit that the incumbent does not use by the incumbent's nearest row,
because the refit may have moved any founder. An accepted addition is followed by one deletion
round from the larger panel, which must reject; otherwise the pass closes a count cycle. Additions
stop at a rejection, a skipped round, a cycle or twice the incoming founder count (a reported
limit, `assembly.founder_refinement.count.COUNT_LIMIT_MULTIPLE = 1`).

**Count cycles.** A cycle is resolved to the smaller count (the deletion round stands) with the
better-F of its two panels at that count (the incumbent and the deletion refit). Both decisions are
flagged ambiguous, and the cycle is undecided when the check round's call is ambiguous and the pass
had reached the smaller count by a decided deletion from the larger one (not by an addition, and not
by deleting a founder without a private cell, which is not a decision).

**Panel at the decided count.** The decisions choose K; the panel kept at K is the best-F converged
fit computed at K during the pass. It differs from the last panel only when the pass returned to a
count it had held with a better panel. The record then keeps the replacement and the restored
panel's adjacent-count comparisons: its recorded decisions and the count rule (margin,
leave-one-sample-out margin, band) against the best-F fits at K - 1 and K + 1. The call is flagged
ambiguous when the restored panel fails its own deletion test (its recorded deletion round accepted
a deletion, or without one, the best-F fit at K - 1 is preferred to it). This is a two-stage policy
(the count decisions choose K, the fixed-K objective chooses the panel), so a restored panel can
fail that test.

**Count call.** A component's call is ambiguous when any decision's band contains zero, the pass
closed a count cycle, or a restored panel fails its own deletion test. Levels before the last make
deletion decisions only; their K is the working set of the next join. The last level's calls are the
components' count calls in `assembly/final` and in `founder_counts.tsv`.

## Bounded continuation and release (2.3)

A selected fixed-K fit tie or tied count-choice panel can affect later deletion and
addition decisions even if the returned fit has another label. Up to sixteen complete
count-pass histories continue these choices from the original input, seed and count state. Cached fits are reused only when
both their label and input panel agree. Terminal incoming join panels share this same
sixteen-leaf budget, rather than each receiving a separate allowance. They use the existing common
complete-site objective and informative cohort. The ordinary count bound remains twice
each incoming root's founder count; it is not enlarged to obtain a preferred outcome.

Completed same-K descendants are compared on the final F objective. Converged identical
panels collapse; conflicting supported counts remain explicit, with smaller K then exact
minimum state changes as the representative prior. An intermediate tie on discarded search
ancestry is diagnostic history, not a current final tie. Count-choice continuations
replay the original history with that decision prefix; an intermediate panel is never
restarted with a fresh count or search budget. A family exceeding the joint history
budget retains its source panels and unresolved status, with no order-selected prefix
represented as a completed family. A partial-span incoming family
without a justified correspondence is likewise not treated as a complete final panel.

Before count-margin eligibility, every already observed fixed-K endpoint in the best F
class is tested against its original deletion/copy reference and incumbent. Identical
endpoint panels from different origins retain their own tests, sample masks and bootstrap
streams. Only eligible count outcomes continue; rejected companions cannot reappear
through an earlier direction-tie family. Count records retain the complete endpoint table
and eligibility decisions.

Nearest-row sequence shadows use the separate complete-family computational budget
`CountRuleConfig.shadow_combinations` (currently 2048). Overflow retains the incumbent's
own evaluated verdict and records `count_unverified` with cause, scope and domain size;
it does not supply evidence for a count change. Every deletion still needs an evaluated
count verdict. Only genuinely opposed evaluated verdicts use the smaller-K prior.

`ComponentRefinement.release_alternatives` distinguishes terminal families from these
uncontinued source histories and from computational `count_unverified` records. The assembled sidecar is persisted with each level and the
final product and survives checkpoint reuse. Export uses it to split founder columns and
tracks where evaluated final alternatives disagree in continuity and withhold unmatched
evaluated final local content. Unique exact row correspondence on invariant path anchors
allows cell-wise agreement to be retained. In unequal-K terminal families, full paths
occurring exactly once in every evaluated member preserve known prepared content; the
remaining unmatched paths use conservative local-row masking. Neither rule removes
continuity cuts or withholding required by another family. Source histories without established final correspondence
retain their evaluated source panels, local disagreement intervals and priors in a
`continuation_unverified` record. The continuation flag itself and all computational `count_unverified` records
annotate the release without withholding calls. An evaluated biological family whose
continuation exceeded the joint/search budget can separately supply a local content meet
inside its complete prior-eligible tie class, using exact prepared-row provenance; see
[assembly release semantics](assembly.md#live-alternatives-and-founder-release-23).
The records expose branch evaluations, cached fits, branch seconds and selected history.

Before retaining a computational flag, refinement may prove the entire shadow domain
with a block-factor dynamic programme. Each block keeps its local shadow choice fixed;
the recurrence maximizes outgoing pair-state messages over all local choices before
continuing to the next block. Its per-sample envelope covers every global shadow, with
the original evidence mask and switch penalty. A conservative effective-carrier bound
must establish that the total-gain test determines the point margin, leave-one-out
margin and every seeded bootstrap replicate under every ordering of the informative
samples. For each original draw, sorted sample multiplicities are paired against
sorted per-sample sequence-minus-total lower gains to bound the worst permutation.
The proof also bounds every reduction order for point and leave-one-out sums and
requires a positive normal gain under every permitted pairing. Outward numerical
bounds cover reduction error and the effective-carrier penalty. This only determines
whether verification is proved: the original bootstrap draws, seeds, count decisions
and score arrays are unchanged. A proof valid only for the current sample ordering
cannot remove the computational flag. A successful proof leaves the evaluated
incumbent decision and all sequence diagnostics unchanged. The record distinguishes
`count_verified_by_bound` from enumeration: `complete=false`, `evaluated=1`, with the
uncomputed sequence diagnostics explicitly identified. Failure to prove equality leaves
`count_unverified` intact and never supplies evidence for a count change.

This optional computation requires each local product to fit the existing shadow
budget, at most four ordinary passes of weighted site work, the existing pair-table
memory cap, and immediate memory admission. It shares the existing workspace and CPU
budget. Proof attempts, reasons, time and transient bytes are recorded separately.
Joins retain the complete-family/unknown policy above; this proof shortcut is used only
in founder refinement.

Continuations with multiple incoming whole-span panels may reuse exact window solutions
within the component. Reuse requires identical ordered window rows, both normalized
boundary messages, convention and frontier limit on the same immutable evidence and row
domains. The optional cache reserves at most 256 MiB through the run's memory governor
and is omitted when memory is unavailable. Eviction or omitted caching changes execution
cost only; proposals, scientific diagnostics, count decisions and released results are
unchanged. The ordinary per-direction proposal cache remains in use.

## Records and census

| Key (under `assembly/l<k>/refine/`) | Content |
|---|---|
| `<n:05d>` | component n: the refined component (row paths and count call) and its refinement record: K before and after, F of the incoming and returned panels, changed cells, bin size, coarse units, cost, every count decision (bounds, ranking, refitted candidates and their F, the count rule's record, shadow cells, cycle flags), every converged fit (both searches' F, sweeps, convergence, largest frontier, fallbacks, accepted moves), the best-F replacement, weak label junctions and wall time |
| `<n:05d>/fit/<label>` | resumable fit states of components with more than `assembly.founder_refinement.records.RECORD_FITS_ABOVE_BLOCKS = 100` prepared blocks: the fit (`<fit>.done`), each search (`<fit>.fwd.done`, `<fit>.rev.done`, with its sweep history stored once) and each sweep (`<fit>.<fwd or rev>.sweep<s>`); labels `start`, `r<i>.del<f>`, `r<i>.add<f>` |
| `level` | the level summary; its presence marks the level complete |

Each component is recorded as soon as it completes and a recorded level is decoded instead of
computed, so an interrupted level resumes from its completed components (and long components from
their last recorded sweep). Every level prints one census line: fits, fits at the sweep limit, the
largest sweep count, the largest window frontier and its memory fallbacks, count-limit hits, count
cycles (undecided ones), ambiguous count calls, best-F replacements, weak label junctions,
deletion rounds with founders without sequence, calls decided by the sequence part, and kept
founders failing the sequence test. A weak label junction is a run of junctions whose best suffix
exchange costs less than one switch penalty
(`assembly.founder_refinement.search.WEAK_JUNCTION_MARGIN = 20.0`): a diagnostic of weakly
determined founder labels, never used downstream.

## Implementation and cost

| Module | Role |
|---|---|
| `assembly/founder_refinement/stage.py` | `refine_level`: component scheduling, surrogate bins, L1 context, count calls, census |
| `assembly/founder_refinement/workspace.py` | a component's fixed inputs: evidence view, complete-site mask, log evidence, surrogate emissions |
| `assembly/founder_refinement/objective.py`, `assembly/founder_refinement/messages.py` | F, the shared max-plus site steps and the exact block messages |
| `assembly/founder_refinement/gains.py` | exact gains of content moves and suffix exchanges |
| `assembly/founder_refinement/acceptance.py` | the compiled Gauss-Seidel acceptance pass |
| `assembly/founder_refinement/proposals.py` | window programmes, their frontier cap and memory budgets |
| `assembly/founder_refinement/guard.py` | the coarse-move guard |
| `assembly/founder_refinement/search.py` | converged fits (best of two) |
| `assembly/founder_refinement/count.py` | the count pass and its optimistic bounds |
| `assembly/founder_refinement/records.py` | records and readers |

**Parallel structure.** The components of a level are independent. They run in a pool of worker
processes that read the chromosome's float32 evidence from shared memory, largest component first,
with as many workers as the CPU budget and the available memory allow (by an estimate per
component of its size); when only one worker fits, they run in the coordinating process. Inside a component,
a fit's two searches, a round's refits and an addition round's copy responses run concurrently,
each with its share of the live thread budget (re-read at numerical boundaries, so the work still
running takes up the threads that finished work frees). Within a search, window programmes run one
thread each, as many at once as the search's share allows; messages and per-sample scores run over
at most eight consecutive sample ranges; the acceptance pass runs as successive compiled calls of 16
junctions. Concurrent work is admitted by byte estimates. Selection order never depends on
completion order, and results do not depend on thread counts, worker counts or memory pacing.

**Complexity.** K is the working founder count, Q = K(K+1)/2, N the samples, L the component's
sites, B its prepared blocks, R the local rows per block offered to a founder, b the bins per window
unit and F the frontier of a window programme. These are work factors, not biological constants.

| Work unit | Work |
|---|---|
| One full score, or the forward and backward messages of a sweep | O(N L K^2) (Potts transitions are O(Q) per site) |
| Exact score of one proposal on its l sites | O(N l K) for samples whose leading state avoids the moved founder, O(l K^2) for each other sample |
| All-junction suffix exchanges, per sweep | O(B N (K^2 log K + K^3)) |
| Window programmes of all founders, per sweep | O(B N (b^2 K^3 + b K^2 F R)) plus dominance tests |
| Guard traceback, per guarded coarse candidate | O(width x affected samples x K^2) |
| One count decision | min(K, 9) converged fits, plus, with more than 9 candidates, the ranking: K unrefitted deletion scores, O(N L K^3), or K copy responses, O(B N b K^2 F R + N L K^3) |

A count step that changes K by d costs at most 9 (d + 1) converged fits (and one check round per
accepted addition); every fit is at most twice the work of one search. The number of sweeps per fit
is data-dependent (100 is a reported limit). Apart from the window frontier, every term is at most
cubic in K. The frontier F is data-dependent: in the worst case it grows as R^W for a window of W
units (W = 8 fine blocks or 4 coarse spans), which the box bound, dominance pruning, the restart and
the fixed frontier cap keep in check; the largest frontier and every fallback are reported.

**Memory.** Per search: O(B N Q) float64 message arrays (two, and six with the coarse-move guard:
switch counts and the guard's forward history), plus the boundaries the proposal cache keeps. Per
component: the float64 log evidence of its complete sites (24 N L bytes) and a reversed copy for
reversed passes, and the surrogate emissions (8 N sum_b R_b^2 bins_b bytes). Per window programme:
up to four times its frontier of 8 N (Q + K + bins + 2)-byte entries, within its current budget. Guard
tracebacks allocate per affected sample an events array and a site-by-state traceback over a window
that doubles until the paths with and without the move agree (at worst the whole component).

**Determinism.** Every kernel uses the same inline max-plus site steps, so message updates, scores
and candidate gains agree bitwise; per-sample values are summed in sample order; refits and seeds
are evaluated in candidate order; bootstrap streams are seeded from the component's span and the
decision's round.

## Interpretation and limitations

- A converged fit is a certified local optimum of F under its move set and the surrogate's
  proposals, not a global optimum; a different start or move set can reach a better panel.
- With more than nine candidates, the ranking (unrefitted deletion margins, copy best responses)
  chooses which candidates are refitted, so a deletion or addition that only another candidate's
  refit would show can be missed.
- K is the count the count rule supports on these samples, not the number of biological founders.
  Rare founders carried by few samples may not pass the effective-carrier test; absent local alleles,
  unsampled ancestry, weak founder representation, founders missing from the local panels and
  long-range mislinking remain limitations. Ambiguous calls mark counts the data do not separate.
- F counts only sites where every local row of the block is called. With more local rows per block
  (larger K) that complete fraction shrinks under the same per-cell missingness, so F reads less of
  the data at large K.
- More likelihood, or more called alleles, does not by itself establish a better biological
  reconstruction.
- A window that reaches the frontier cap keeps its incumbent; every such fallback is reported.
