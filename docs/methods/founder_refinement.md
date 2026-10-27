# Founder refinement

Founder refinement (`assembly/founder_refinement/`) runs after every executed level of the final
[assembly](assembly.md), never in the context assemblies of local selection. It re-chooses each
phase component's founders as paths of the original prepared local rows (one row per prepared
block; the rows' alleles and evidence masks stay fixed), fitting them to a certified local optimum
of a cohort likelihood objective F, and it decides the component's founder count K between converged
fits with the pipeline's [founder-count rule](assembly.md#founder-count-rule). Every level may
delete founders; only the last level (the final count step) may also add them. After the count is
decided, the decided panel is [polished](#polish) with a wider move set at the same K. The refined
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
- the contig's genetic map (the objective's switch costs; `[genetic_map]`, see
  [genetic maps](../genetic_maps.md));
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
F_s is a robust maximum-path score of sample s's diplotype: the Viterbi recursion of a per-homologue
founder-switching model, with clipped and centred site emissions (below) and the model's normalisers, so it is a
log-probability-like score rather than a literal log probability (`assembly/founder_refinement/objective.py`):

- each homologue follows its own chain over the K founders. Between consecutive informative sites of
  the sample it keeps its founder with probability 1 - rho (1 - 1/K) and moves to any given other
  founder with probability rho / K, where rho = 1 - exp(-h) and h is the reset hazard of the gap:
  `AssemblyConfig.refinement.reset_rate = 1.0` per Morgan per homologue times the genetic-map length of
  every physical edge the sample crosses (each edge at least -log(1 - floor), with
  `AssemblyConfig.refinement.map_probability_floor = 1e-15`, so sites at one map position still cost a
  finite switch);
- relative to staying, a homologue that changes founder pays w = log(1 - rho (1 - 1/K)) - log(rho / K)
  >= 0. Both homologues share rho, so the ordered K x K chain reduces exactly to the Q = K(K+1)/2
  unordered founder pairs: U'{a,b} = max(U{a,b}, max(R[a], R[b]) - w, G - 2w) + e, with
  R[x] = max_y U{x,y} and G = max U, which costs O(Q + K) per site;
- the panel-independent terms (-2 log K at the sample's first informative site and
  2 log(1 - rho (1 - 1/K)) at every later one) form the sample's normaliser. They cancel at a fixed K
  and are included wherever fits of different K are compared. The uniform start charges every sample, carrier or
  not, 2 log((K + 1) / K) per added founder (a total that grows with N); it can delete a rare true founder at a small
  component (in the f3 simulation, one carried by 2 of 112 samples at L1). A free start (the starting pair
  profiled out) was tested and not adopted: it recovered such L1 founders, but the whole-pipeline outcome was mixed
  (std better by 565 wrong-or-missing alleles, backcross worse by 2,751 through a changed L2 join);
- a site emits max(log(0.99 p + 0.01/3), -2) - log(1/3) for the state's pair class, with p the
  normalised genotype likelihoods of the sample. The pair class is the dosage when both founders'
  local rows are called; otherwise it is the predictive class of the uncalled allele(s) under a
  uniform allele prior: one uncalled allele beside a called one gives (p0 + p1)/2 or (p1 + p2)/2, two
  different uncalled rows (p0 + 2 p1 + p2)/4, and the same uncalled row twice (one latent allele)
  (p0 + p2)/2;
- the sites are the component's kept discovery sites with at least one called local row in their
  prepared block (the assembly's usable sites, also the window surrogate's mask), where the sample's
  evidence is not exactly uniform. The mask is fixed across candidate panels, so a panel cannot raise
  its score by hiding sites. Until 2.7, F read only sites where every local row was called; that
  fraction shrinks with more local rows (at K = 30, 18-25% of a test component's sites have an
  uncalled local row).

2.6's objective charged a uniform 20 nats per diplotype change. The switch costs are now those of an
explicit switching process, per sample and gap; the binned window surrogate keeps the uniform
`assembly.founder_refinement.objective.SWITCH_PENALTY = 20.0` (it only proposes moves). F is an internal
assembly objective, not the downstream homologue painter, a phase posterior, a genealogy model or a
recombination-map estimate. Count decisions charge each
founder the cost c = s x (mean prepared-panel sites / 200) x S x B of the count rule, for B prepared
blocks and the S informative samples of the component: samples with evidence at one of its sites.
Every other sample (samples excluded from founder construction, whose evidence is uniform, or without a read at
one of its sites) scores exactly 0 for every panel, so it enters neither the cost nor the bootstrap population.
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
   applied while its exact gain exceeds `assembly.founder_refinement.acceptance.TOL = 0.001953125` (2^-9, release
   2.13: at least 16 times the typical motion of refinement contrasts under 1e-6 relative GL noise, 9e-5 nats at p99
   over about 4,300 matched-history fit contrasts of jitter runs, a design scale and not a worst-case certificate;
   1e-4 before); then
   the proposals anchored at the junction (one founder's rows on the blocks where a proposal differs
   from the current panel) are re-scored exactly against the current panel and applied best first
   (earliest on ties) while their gain exceeds the same tolerance; coarse proposals must also pass
   the [guard](#coarse-move-guard). After `assembly.founder_refinement.acceptance.CONTENT_FULL_ROUNDS = 4`
   acceptances at one junction, the remaining proposals of the last full ranking are re-scored one at a
   time in that order (each applied while its own gain exceeds the tolerance), and at most
   `assembly.founder_refinement.acceptance.EXCHANGE_ROUNDS = 4` suffix exchanges are applied per junction and
   pass. Both bounds keep a pass at most cubic in K; a binding bound is counted (`round_cap`,
   `exchange_cap`) and warned, and neither changes the fixed-point test, because a pass that accepts
   nothing scores every proposal once;
4. recomputes F, which must equal the start of the sweep plus the accepted gains within
   1e-9 max(1, |F|) + 1e-6 per accepted move; a larger deviation stops the run with an error.

A sweep that accepts nothing certifies the fit: every proposal of that sweep was computed against
the final panel and every offered move was scored exactly against it, so no offered move gains more than
the tolerance (under the guard and the proposals' screening); the panel is a local optimum of F under
these offers, not under every move of the families. F is non-convex and no global optimum is claimed. The window
proposals optimise a binned surrogate, so the certificate is relative to them; acceptance always
scores individual sites. A search that reaches `AssemblyConfig.refinement.sweep_limit = 100` sweeps
is reported (warned, and recorded as not converged), never silently accepted.

**Best of two.** Every fit runs this search twice from the same start, concurrently: once with its
first pass forward and the default content-order convention, once reversed with the reversed
convention (both then alternate; each computes its own first-sweep inputs). Both are deterministic local
searches. The existing F tie classes and count comparison criteria identify the surviving candidates;
among remaining same-K ties the representative uses the exact minimum number of changed homologues over
the samples' maximum-score paths. Both results are recorded.
This applies to every start fit and every count refit; a polish fit runs the forward-first search only, with
the whole budget.

**Lexicographic sweeps.** The guard reads switch costs of maximum-score paths, computed by
lexicographic messages (score, then the lower switch cost among exact score ties), which cost about
three times the plain ones. A guarded search computes them only for sweeps that read them: a sweep
whose predecessor read none runs its acceptance pass on plain messages and, at the first coarse
proposal whose gain exceeds the tolerance, is run again on lexicographic messages from its start
(a search's first sweep starts lexicographic). The plain and lexicographic scores are bitwise equal,
so the decisions are those of a fully lexicographic search.

**Surrogate bins.** The window surrogate cuts each prepared block's sites into bins of
max(b_min, ceil(L_c / 2000)) sites for a component of L_c sites
(`AssemblyConfig.refinement.proposal_max_bins = 2000`) and sums over a bin's discovery sites the
robust emission of the join's binned model: hard dosages where every row of the block is called at
its discovery sites, otherwise the predictive pair classes with unknown alleles marginalised (the same
usable sites as F). At levels 3 and above, b_min is
`AssemblyConfig.refinement.proposal_min_sites_per_bin = 200`; at levels 1 and 2, b_min is
max(200, ceil(chromosome sites / 2000)), shared by all the components of the level, so the many
small early components do not each receive a chromosome's bin budget.

### Window programmes

A window problem maximises the surrogate over one founder's rows on the window's units (prepared
blocks or L1 spans). The solver (`assembly/founder_refinement/proposals.py`) first tries the box bound (every sample choosing the
founder's later rows freely), which certifies windows whose incumbent is already optimal. Otherwise
an exact forward frontier programme runs: a partial path is dropped only when another is at least as
good for every completion (closed-message dominance) or when a valid upper bound on its completions
lies below a feasible value. The answer is canonical, a pure function of the problem: among paths within
2^-14 + 1e-9 |H| of the optimum H (release 2.13; 1e-7 + 1e-9 |H| before), the incumbent if it is one, else the
lexicographically
first path (units left to right, each in the order incumbent row, then ascending row index). A
programme whose frontier passes `assembly.founder_refinement.proposals.EXPLORE_FRONTIER = 256`
partial paths restarts with tighter pruning (Lagrangian suffix bounds and the best feasible value of
the incumbent, a 16-wide beam and the dual's decoded paths); pruning keeps every path within the
margin of a feasible value, so the optimum and the canonical answer are unchanged. The restarted
programme keeps at most `assembly.founder_refinement.proposals.BEAM_CAP = 512` children per unit (2.8:
a polynomial bound in place of an unbounded frontier); where the cap binds the answer is approximate,
every truncated unit is counted (`beam_truncations` in the sweep, fit and level work counters) and
warned, and a main programme that loses its feasible witness that way keeps the incumbent
(`BEAM_LIMIT`, a reported fallback).

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
(`assembly/founder_refinement/guard.py`): a coarse move that lowers the genotype fit E = F + C (C the
switch cost of the samples' maximum-score paths, the lowest among exact score ties) is refused unless
its F gain exceeds the acceptance tolerance once every saved switch at a shared one-way junction is
uncredited, dF - C_shared > TOL. C_shared sums over samples the smaller of the switch cost the sample
saves and the cost of its dropped switch events at shared one-way junctions (w per changed
homologue, 2w for an event changing both, at the event's site; for the better pairing of each
transition: a conservative upper bound on the shared cost). A dropped event is a founder transition
u -> v of one homologue that the move eliminates; it is one-way unless it leaves the edited founder
and the same path later rejoins it (an excursion, the signature of a row that is wrong there), and
shared when another sample's one-way dropped event is the same transition with an overlapping
evidence interval (the stretch between the sample's nearest informative sites where u and v
differ). Moves that do not lower E, and moves whose gain comes from unshared junctions or
excursions, pass. The guard applies only where coarse moves exist (levels 2 and above). It can
withhold a true move; it is not a confidence threshold or a proof of biological correctness.

## Polish

The count pass decides K with fits of the search above. The decided panel, and each of its same-K
equal-F alternatives, is then refitted once more (`AssemblyConfig.refinement.polish = True`; fit labels
`polish`, `polish.alt<i>`) by one forward-first search with three more move families, tried in this order
at every sweep that accepts nothing; when one accepts, ordinary sweeps resume:

1. **Beam escape.** The window-beam proposer (`assembly/founder_refinement/beam.py`) solves every fine
   and coarse window for every founder on the window programmes' binned emissions, but with
   map-derived per-sample reset costs between bins and the surrogate's own flank messages. It keeps the
   `assembly.founder_refinement.beam.BEAM_WIDTH = 64` best partial paths per unit (an approximate
   search: cut windows are counted, `escape_beam_cut` and `escape_beam_binding`, and warned) and offers
   each window's best path that differs from the incumbent when its surrogate gain exceeds the tie
   quantum. The beam ranks a unit's children by their bounds rounded in quanta relative to the unit's best bound
   (release 2.13; before, on a fixed grid of the totals, which a 1e-6 GL perturbation shifts in common by
   several quanta, so near-equal children traded places at its rounding boundaries); a later distinct path
   replaces the best offer when it is better by half a quantum or more. Cuts at the width still separate
   children whose rounded keys differ, so near-equal children at a cut can still trade places. One exact acceptance pass over these offers follows (coarse offers pass the guard where coarse
   moves are guarded), and its gain is verified by full rescoring.
2. **Long repair.** Beam offers on windows of `assembly.founder_refinement.search.LONG_WINDOW = 64` blocks
   (every `LONG_STRIDE = 16`; beam width `assembly.founder_refinement.search.LONG_BEAM_WIDTH = 16`, cut
   windows counted as `long_beam_cut` and `long_beam_binding` and warned). Each offer, and every variant of
   it whose rows on one stretch of `LONG_CORRECTION = 8` blocks (starting inside the offer's changed span,
   within the offer's window) are replaced by the incumbent's rows or those of one of the `LONG_DONORS = 2`
   founders that agree with the offer on most blocks of that span (one private correction), is scored
   exactly as a one-founder move. Per offer, the best variant whose gain exceeds the tolerance (the
   content-smallest among those within the refinement quantum 2^-9 of the best, release 2.13) enters one
   acceptance pass as a coarse move, so the shared-junction guard applies where coarse moves are guarded
   (components with an L1 context, levels 2 and above); a guard rejection leaves the offer's other variants
   untried. A long surrogate optimum can be exactly F-negative only over a short stretch; the ordinary
   8-block windows do not reach such a move in one step (on the stress diagnosis component its 8-block
   pieces each lose 34-105 nats).
3. **Bridges.** For every fine window, the offered change of largest exact gain (one precursor per window,
   from the escape's offers) is applied privately and every founder-pair suffix exchange at every cut
   inside the window is scored exactly against it; the best combined gain above the tolerance is applied
   after full rescoring. Each of these choices takes the content-smallest candidate in the band
   [best - q, best], q the refinement quantum 2^-9 (release 2.13; a maximum band, not the chained tie
   classes; before, exact ties only, so near-equal gains were chosen by their last bits and a 1e-6 GL
   perturbation sent the polish along another path). A bridge picks its exchange before testing the
   combined gain against the tolerance, so a tied exchange can leave the window without an offer. Bridges are not subject to the shared-junction guard.

The best polished fit (tie classes of F) and its equal-F family then follow the count pass's own
terminal rule (the fewest exact-optimal switches). The polish never changes K; the count decisions,
their bands and the best-F restoration are those of the count pass. Each escape runs on the panel's
canonical founder order (rows sorted by content) and maps its result back, so it depends on the panel
only up to founder labels. The refinement record keeps the polish's status, the count pass's final fit and
F, the polished F and gain. The count decisions, their bands and the best-F restoration were taken on the
pre-polish panel and are not re-evaluated on the polished one (`count_reevaluated` false). A polish that
reaches the component's window work bound keeps the count pass's panel unpolished (status
`window_work_bound`, warned): the polish is a separate stage after a complete count pass.

The count pass uses the ordinary search only: run in every count refit, the escape cost about five
times 2.6's refinement time on the validation set, while as a polish it keeps those gains within
twice 2.6's time.

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
rule. Its sequence part uses the [symmetric null](#symmetric-sequence-null); the shadow is scored exactly (refinement takes no near-identity
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
acceptance pass) are refitted. The best-F refit is decided against the incumbent with the same
[symmetric null](#symmetric-sequence-null) as a deletion (the refit's own death class), so an accepted
addition needs no check deletion round (2.7 ran one, against the incumbent's nearest rows). Additions stop at a
rejection, a skipped round or twice the incoming founder count (a reported limit,
`assembly.founder_refinement.count.COUNT_LIMIT_MULTIPLE = 1`).

### Symmetric sequence null

A decision between a larger panel and a smaller one compares the larger
panel with its own canonical death candidates: the members of the best tie class of its unrefitted
one-founder deletions (founders without a private cell first, otherwise all). Every member is
evaluated; the larger panel is accepted only if every member's test accepts it, a disagreement or a
member whose nearest-shadow family is unresolved gives the smaller K with an ambiguous call, and the
record keeps every member's verdict. The same two endpoints therefore get the same decision whichever
of them was the incumbent.

The optimistic deletion and addition bounds use the target K's switch costs and add its normalisers.

**Count cycles.** Until 2.7 an accepted addition was checked by a deletion round from the larger panel,
and a check that accepted the deletion closed a count cycle. With the symmetric null both directions
decide the same two endpoints identically, so 2.8 runs no check round and no cycle arises; the records
keep their `count_cycle` and `cycle_undecided` fields, always false.

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
sixteen-leaf budget, rather than each receiving a separate allowance. They use the component's common
site mask, objective and informative cohort. The ordinary count bound remains twice
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
| `<n:05d>/fit/<label>` | resumable fit states of components with more than `assembly.founder_refinement.records.RECORD_FITS_ABOVE_BLOCKS = 100` prepared blocks: the fit (`<fit>.done`), each search (`<fit>.fwd.done`, `<fit>.rev.done`, with its sweep history stored once) and each sweep (`<fit>.<fwd or rev>.sweep<s>`); labels `start`, `r<i>.del<f>`, `r<i>.add<f>`, `polish`, `polish.alt<i>` |
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
| `assembly/founder_refinement/workspace.py` | a component's fixed inputs: evidence view, site mask, coded leaves, log evidence of the pair codes, map hazards, switch costs and normalisers, surrogate emissions |
| `assembly/founder_refinement/objective.py`, `assembly/founder_refinement/messages.py` | F, the shared max-plus site steps and the exact block messages |
| `assembly/founder_refinement/gains.py` | exact gains of content moves and suffix exchanges |
| `assembly/founder_refinement/acceptance.py` | the compiled Gauss-Seidel acceptance pass |
| `assembly/founder_refinement/proposals.py` | window programmes, their frontier cap and memory budgets |
| `assembly/founder_refinement/guard.py` | the coarse-move guard |
| `assembly/founder_refinement/search.py` | converged fits (best of two), lexicographic sweeps, the polish's escape, long repair and bridges |
| `assembly/founder_refinement/beam.py` | the polish's window-beam proposer |
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
unit, F the frontier of a window programme and h = 64 the polish beam's width. These are work
factors, not biological constants.

| Work unit | Work |
|---|---|
| One full score, or the forward and backward messages of a sweep | O(N L (Q + K)) (per-homologue transitions are O(Q + K) per site) |
| Exact score of one proposal on its l sites | O(N l K^2) |
| All-junction suffix exchanges, per sweep | O(B N K^3) (at most five scans per junction) |
| Content proposals at one junction | at most five evaluations per proposal (O(K) proposals) |
| Window programmes of all founders, per sweep | O(B N (b^2 K^3 + b K^2 F R)) plus dominance tests; F is at most 512 per unit after a restart |
| Guard traceback, per guarded coarse candidate | O(width x affected samples x K^2) |
| Polish beam, per window and founder | O(W N (b^2 Q + h b (Q + R (K + b)) + h^2 Q)) for W units |
| Long repair, per polish fixed point | O(B K / 16) offers, each with O(48) variants of O(N x 64 blocks x K^2) |
| One count decision | min(K, 9) converged fits, more when a tie class straddles the cut (the whole class is kept, up to K), the death class's shadow decisions (at most K), plus, with more than 9 candidates, the ranking: K unrefitted deletion scores, O(N L K^3), or K copy responses |

A count step that changes K by d costs about 9 (d + 1) converged fits, up to K (d + 1) when tie
classes straddle the shortlist cut, plus count continuations; every fit is at most twice the work of one
search, and the polish adds one single-direction fit per decided panel and equal-F alternative. The
number of sweeps per fit is data-dependent (100 is a reported limit). Every per-sweep term is at most
cubic in K for R = O(K) (the bound is per sweep; the number of sweeps, refits, rounds and continuations
are data-dependent factors); the window frontier, which was data-dependent up to R^W, is capped by the
restart's 512 children per unit, and every binding cap is reported.

**Memory.** Per search: O(B N Q) message arrays (two in plain sweeps, four in lexicographic ones), plus two carried-history arrays at every16th block boundary and the
boundaries the proposal cache keeps. Values and lexicographic switch costs are float64. Per
component: the float64 log evidence of the seven pair codes (56 N L bytes) and a reversed copy for
reversed passes, the coded leaves (int16, both orientations), the surrogate emissions
(8 N sum_b [R_b(R_b+1)/2] bins_b bytes) and, with the polish, the beam's per-bin table
(8 N J (4 + R_max(R_max+1)/2) bytes for J bins); the polish's escape messages, bridge and long-repair
batches and beam scratch are reserved when they run. Per window programme:
up to four times its frontier of 8 N (Q + K + bins + 2)-byte entries, within its current budget. Guard
tracebacks allocate per affected sample an events array and a site-by-state traceback over a window
that doubles until the paths with and without the move agree (at worst the whole component).

**Determinism.** Kernels preserve the same max-plus model and released decisions; mathematically
equivalent floating-point reduction orders need not produce identical bits; per-sample values are summed in sample order; refits and seeds
are evaluated in candidate order; bootstrap streams are seeded from the component's span and the
decision's round.

## Interpretation and limitations

- A converged fit is a local optimum of F under its move set and the surrogate's proposals, not a
  global optimum; a different start or move set can reach a better panel. The polished panel has no
  offered improvement under a wider move set; the beams that propose those moves are approximate by design,
  and only one variant per long offer and one precursor per bridge window are tried.
- The count decisions compare fits of the ordinary search; only the decided panel (and its equal-F
  alternatives) is polished, so a count call does not see what the polish might have found at the
  other count.
- With more than nine candidates, the ranking (unrefitted deletion margins, copy best responses)
  chooses which candidates are refitted, so a deletion or addition that only another candidate's
  refit would show can be missed.
- K is the count the count rule supports on these samples, not the number of biological founders.
  Rare founders carried by few samples may not pass the effective-carrier test; absent local alleles,
  unsampled ancestry, weak founder representation, founders missing from the local panels and
  long-range mislinking remain limitations. Ambiguous calls mark counts the data do not separate.
- F reads a site where some local rows are uncalled through the predictive classes of the uncalled
  alleles under a uniform allele prior, per observation; it does not infer one shared value of an
  uncalled founder allele across samples. Sites where every local row is uncalled are not read.
- The switch costs assume crossovers as a reset process along the genetic map at one reset per Morgan
  per homologue; F is a maximum-path score, not a marginal likelihood, and it is not a recombination-map
  estimate.
- More likelihood, or more called alleles, does not by itself establish a better biological
  reconstruction.
- A window that reaches the frontier cap or loses its witness to the beam cap keeps its incumbent;
  every such fallback is reported.


Release2.7 packs symmetric emissions and retains carried guard history every16blocks,
reconstructing intermediate boundaries exactly. Full-width traceback and event-matching
workspace is included in admission. This reduces resident memory but can add guard runtime.
See [assembly work and memory](assembly_performance.md) for diagnostics, measured scope and
the distinction between a frontier memory cap and a cubic bound on actual window work.
