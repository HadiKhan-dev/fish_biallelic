# Family phase

Family phase (stage `family_phase`, entry point `family_phase.stage.run`) gives every sample final genotypes, phase,
phase sets and released uncertainty for every chromosome, conditioned on the supported Tier-B parent edges released by
[pedigree inference](pedigree.md). Each painting component gets one generative model: every homologue is a mosaic of
the [painting](painting.md)'s founder labels, and a sample's homologue copies a released parent's homologue through a
meiosis chain. The stage solves the model by exact block coordinate ascent under both tie conventions. The ascent
ends settled within the gain quantum (no block's exact optimum, given the others, improves the objective by more than
it), or else at a recorded sweep cap. It then
releases each sample from the exact posterior of that sample's block, given the final state of all other samples,
and keeps only what the two solves agree on. It runs genome-wide after the [founder refit](founder_refit.md), one
chromosome per worker, and
[recombination](recombination.md) follows the meiosis chains it releases. Family phase does not modify the pedigree, the painting or
any founder panel, and nothing feeds back upstream.

Release 3.0 rebuilt the stage from scratch. The 2.x scaffold, family MAP solver over genotype potentials, root-frame
repair and phase polish are gone. The product keeps the 2.x record layout, so export and evaluation read it
unchanged.

## Inputs and outputs

| | record or setting | content |
| --- | --- | --- |
| reads | `painting` | per component: states (trajectory classes plus BACKGROUND), bins, the estimated label transitions, the optional pair-emission cache, released chunks and local allele keys (for tie boundaries), eligibility (for coverage) |
| reads | `founder_refit` | the refit panels' inference alleles (the released alleles of each label) |
| reads | `evidence` | site positions; genotype likelihoods and observed masks only when a component's emission cache is absent |
| reads | `ParentEdges` | the supported Tier-B parent → child edges (parent, child, slot 0/1 for the child's Parent1/Parent2 column) |
| reads | the contig's genetic map | meiosis switch probabilities (Haldane) between bin centres |
| reads | `[family_phase] protected_intervals` | optional breakpoint TSV: intervals across which meiosis chains reset; a reversed segment resets at both ends |
| writes | `identity/family_phase` | `FamilyPhaseConfig` (protected intervals included), the identities of `painting`, `founder_refit` and `evidence`, the edges' identity, the map's digest, code identity |
| writes | `family_phase` | the `FamilyPhaseProduct`; run record `family_phase` (summaries) and `complete/family_phase` |
| publishes | `OUTPUT/family_phase/summary.csv` | one row per contig |

The other `FamilyPhaseConfig` fields are internal model constants ([configuration reference](../configuration.md)).
Before phasing a contig the stage checks that its painting was computed from the current evidence record and that
the painting and pedigree sample axes agree.

## Model

The model runs per painting component, so label sets are independent between components. Its bins are the painting
bins (about 100 selected sites each), and its labels are the painting states: C trajectory classes plus BACKGROUND,
S = C + 1.

- **Homologues.** Every sample has homologues 0 and 1.
  - **Linked homologue.** Homologue k of a sample with a released slot-k parent is linked. Its source chain V takes
    values in {0, 1, ⊥}:
    - 0 or 1 means a copy of that parental homologue, whose label it takes;
    - between bin centres the chain switches 0 ↔ 1 with the Haldane fraction of the genetic map's interval, and the
      fraction is 1/2 across protected intervals, where transmission resets (reference-order breakpoints, and both
      ends of reversed segments);
    - ⊥ means not a copy: a painting or panel error, a conversion or mutation, or a false edge. The chain enters ⊥
      with α = 1 − exp(−r Δ) and leaves it with β = 1 − exp(−Δ / m), where Δ is the bp distance between bin centres,
      r = `not_copy_rate_per_bp` and m = `not_copy_mean_bp`.
    - Entering ⊥ draws a uniform label, and leaving it picks a parental homologue with probability 1/2 each. While
      in ⊥, the label follows the free label process.
    - At bin 0 a linked homologue is a copy of either parental homologue with probability (1 − π)/2 each, or ⊥ with
      a given label with probability π/S. Here π = r m / (1 + r m) is the ⊥ occupancy of the continuous-rate
      process, which need not equal the bin-level chain's α/(α + β).
  - **Free homologue.** A homologue with no released parent in its slot is a label path under the painting's
    copying process. At each bin edge it switches label with probability θ_l, the per-track switch of the
    painting's estimated transitions: the painting's pair stay probability is the square of a track's. The
    destination is uniform over the other labels, and θ_l is at most (S − 1)/S.
- **Emission.** A sample's emission in a bin is the painting's tempered log emission of its unordered label pair:
  the painting's own robust genotype-likelihood model, divided by its design effect. The stage takes it from the
  painting's emission cache or, when the cache is absent, recomputes it with the painting kernel. Each sample's reads
  enter once.
- **Objective.** The log posterior J of a configuration (labels and chains) sums every sample's emissions, every
  chain and label transition, and the linked homologues' initial terms. A sample without relatives reduces to the
  painting's decode, without its fidelity states.

## Inference

- **Blocks.** A sample's block is an exact HMM over three components:
  - the source of homologue 0;
  - the source of homologue 1;
  - the gauge g of its children.

  Source states:
  - a linked slot has S + 2 states: two copies, and ⊥ with each label;
  - a free slot has S labels.

  g = 1 reads every child's chain for this sample as flipped from that bin on. One block therefore contains every
  move that relabels this sample's homologues from some bin on, and g's transition cost is exactly the change in
  the children's chain terms.

  The block's emission at a bin is the sample's own emission plus those of every descendant homologue that copies
  it at that bin under the current chains, as functions of the copied label. A descendant that copies both of the
  sample's homologues (inbreeding) contributes its pair emission once.

  Because ⊥ entry draws a uniform label, a change of a parent's labels never alters a descendant's transition
  terms, so the block is the exact conditional of the full objective. The factorised max-product contracts the
  three transition factors one at a time, at O(B · states · (n₀ + n₁ + 2)) per block.
- **Schedule.**
  1. Initialise every sample by its own painting Viterbi: both homologues free, no relatives.
  2. Visit samples with a released parent, ancestors first, without their children's terms; parental origin then
     orients them.
  3. Sweep blocks in pedigree order: ancestors first, within a depth by sample index. A block's path is applied
     only when it raises J by more than `gain_quantum`, so the ascent is monotone and terminates. `max_sweeps` is a
     work cap: a component that reaches it keeps its current, valid state and is recorded as not settled.
  4. Joint moves on the parents' pairings at double label changes (below, "Parents' pairings"), each evaluated by
     its change of J, then the sweeps again, until no move is kept.
- **Local gains.** J is a sum of per-sample terms, and each sample's terms read only its own labels and chains (a
  linked homologue's transitions depend on its source states, not on the copied labels). A block of sample n
  changes only n's labels and chains, its children's chains (the gauge) and the copied labels of its descendants,
  and a pairing move of parent p only p and its descendants. A move's gain is therefore the change in the terms of
  those samples, computed sample by sample, and only their rows are saved and restored; J itself is evaluated once
  per sweep, for the recorded history. The decisions are those of comparing the full objective before and after
  each move, as releases before 3.4 did, except where a gain lies within floating-point rounding of a threshold.
  Full-objective subtraction and the old block rule's addition of `gain_quantum` carry rounding; the local gain
  carries only that of the changed samples' terms. The observed full/local differences depend on the component
  and are not a universal bound. `LabelModel.check_deltas` re-evaluates every move with the full objective and
  records every move where the full-objective rule decides differently (a validation path, off in runs). For
  blocks it compares the literal old rule `J1 > J0 + gain_quantum` with `local_gain > gain_quantum`, rather than
  subtracting the two totals. For D2 pairing moves it retains the old full-difference predicates: keep when
  `J1 - J0 > max(pairing_odds, gain_quantum)` (if keeping moves), and ambiguous when `J1 - J0 > -pairing_odds`.
  Each flip records both decisions; the solve still follows the local gain.
- **Ties and conventions.** Viterbi scores within `tie_quantum` are tied. Ties keep the current state and gauge 0 (a
  stated prior), then the lowest index; the mirrored convention takes the highest index and reverses the within-depth
  order.
  - Every component is solved under both conventions, and the release is their agreement (below). The released
    output therefore does not depend on a convention, and the run's convention reversal (`HAPLOTYPES_TIE_REVERSE`,
    [ties](ties.md)) leaves it unchanged.
  - Exact ties arise where labels are equal in content over a stretch. Which label the ascent keeps does not change
    that sample's own posterior, but it changes what its relatives' posteriors condition on.
- **Exactness.** Exhaustive enumeration on tiny pedigrees checks every block of every sample: founders, full sibs,
  an inbred child of sibs, a backcross and a one-parent child, with ⊥ runs. The block Viterbi reaches the maximum of
  the full objective, and forward-backward matches the enumerated posteriors to about 1e-15.

## Identity junctions and painting ties

A panel row is a trajectory through assembled blocks, not a certified biological homologue. Where a row follows one
lineage, samples from other lineages see label switches. Two cases look alike:

- **Assembly.** Two rows exchange the founders they follow at a block junction.
- **Shared ancestry.** An unsequenced parent's crossover is inherited by several of its children while the rows stay
  with their founders.

Either way, carriers of one of the two labels a and b switch between them at the same bin edge. Carriers of both
keep continuous labels, and whether their content changes homologue there cannot be seen in their emissions. Without
relatives that pin its homologues, the relative phase of such a double carrier across the edge is not identified. A
global relabelling chosen by J is therefore unsafe: on a TroMau-like simulation it accepted shared ancestral
crossovers as exchanges and flipped the phase of a founder sample at every one.

- **Junction candidates** are the bin edges where the free or ⊥ tracks of at least `junction_min_samples` distinct
  samples switch between the same two labels.
- **Withheld joins.** A sample whose two homologues carry labels a and b on both sides of a candidate edge breaks
  its phase set there, unless it is pinned. A count of relatives is no evidence: a relative without reads on one
  flank pins nothing. A sample is pinned only by a linked homologue that copies the same parental homologue on both
  sides of the edge, when that parental homologue is pinned by the parent's own reads. On each flank
  (`junction_pin_bins` bins, cut at the parent's own label changes), the parent's emissions for its pair must beat
  every pair that puts the other junction label on that homologue, whatever the partner, by at least
  `junction_pin_nats`.
  - The partner is free because two rows can be exchanged together, and a pair alternative can then fit the same
    unphased reads.
  - A parent carrying both labels never pins.
  - Children are not used. A child's copy rules out the alternative only through a meiosis switch at exactly that
    edge, whose odds depend on the interval (none across a protected interval) and on uncertainty the children
    share.
  - No relabelling is made.
- **Painting ties.** A sample without a released parent also breaks its sets at painting tie boundaries. These
  are the released chunk boundaries where the painting's track order was set by a tie: the ordered and the swapped
  continuation change equally many candidate rows and allele-visible keys. A linked homologue is ordered by parental
  origin, so painting ties do not affect it.
  - A sample with neither parent nor children (a gauge sample) breaks at every tie.
  - A sample with children and no released parent takes its orientation from its children's meiosis chains on both
    sides of a tie, so the painting's arbitrary track order there does not bind it. It keeps a tie break only where
    that orientation is uncertain beside the tie: where the orientation certainty (below) under its
    children-anchored posterior is below `set_posterior` at the nearest bin with a confidently heterozygous site on
    either side, or where one side has no such bin. Each solve decides for itself, and the agreement keeps a break
    that either solve keeps. A tie that is also a withheld join stays a break. Its calls do not depend on breaks, so
    a bridged tie releases exactly the as-solved join.
  - On the validation simulations (replays of 13 designs and of the similar-founder stress design, and fresh seeds
    of two of them), none of about 3,000 bridged joins scored against truth was wrong. Without the gate, 4 of 1,961
    were wrong (three founders of a full-sib design beside locally equivalent rows, and the parent of the stress
    design), each with an orientation certainty of 0.85-0.9988 on one side.
  - The certainty is conditional on the children's chains at the final state. It cannot see which child recombined
    when the children disagree at the tie, nor an identity exchange that every child follows.
  - The children's meiosis chains do not reset at a bridged tie (below), so [recombination](recombination.md) can
    call a crossover there. On the simulations this added about 1% more calls, 64 of 66 holding a true crossover.
- **Parents' pairings.** Where both homologues of a sample without a released parent change label at one bin edge,
  its own emissions (of unordered pairs) cost the same under both pairings of the old labels with the new: only its
  children decide the pairing. A child that copies the arbitrary pairing absorbs it with a label switch of its other
  homologue, and no single block can undo that, because the better configuration changes the parent and the child
  together.
  - After the block ascent, each solve tries the other pairing at every such edge: the parent's homologue labels are
    swapped from the edge on, its children's chains are left as they are (so they copy the swapped content), and
    every child gets its exact block optimum. The move is kept only when it raises J by more than
    log(`set_posterior` / (1 − `set_posterior`)) = 6.9 nats, the odds at which a set continues; the block sweeps then
    resume, and the passes repeat until none is kept (at most `max_sweeps`). A move that leaves every label and chain
    as it was (for example a homozygous pair on both sides of the edge) is no competing pairing and is ignored.
  - The margin is a working value against one optimized alternative, not a calibrated posterior of the pairing: the
    children are re-solved one block at a time rather than marginalized jointly, and other configurations can
    contribute.
  - A pairing whose alternative is closer than that, in either solve, is not decided by the data: the parent and all
    its children break their sets at that edge, and neither orientation gate bridges these breaks. All children
    break, not only those the alternative relabels: a child can keep its labels under the evaluated alternative
    (crossing over instead) and still have its phase across the edge decided by the pairing.
  - The breaks reach the parent and its children, not further descendants, and their number grows with the size of
    the family.
  - On the validation simulations this occurs once, in the similar-founder stress design, where the child of such a
    parent carried a wrong set of 11,330 sites; it is now broken there (0 error sites).
- **Releases across breaks.** A sample with a released parent or with children keeps its orientation from origin or
  from its children's chains on both sides of a break, but the phase across the break is not released. The
  meiosis chains of every edge involving the sample reset there (below).

## Release

Every sample is released from the exact forward-backward of its own block, with all other samples fixed at the
final state. Label-pair posteriors per bin become ordered allele posteriors per site through the site model
(`kernels.site_posteriors`):

- A label's allele at a site is the common refit inference allele of its class's rows when all of them are called
  and agree; otherwise it is unknown, and BACKGROUND's is unknown. A label pair with an unknown allele contributes to
  no genotype, so its posterior mass leaves the call undecided.
- Given a label pair (x, y) with known alleles, the sample's ordered alleles (u, v) have probability
  (1 - `site_deviation`) f_x(u) f_y(v) + `site_deviation` U(u, v).
  - U is a uniform genotype (ordered weights 1/3, 1/6, 1/6, 1/3), with `site_deviation` = 1e-3: a label error
    shorter than a bin, a conversion or a mutation.
  - f_x is the panel allele of label x with its own uncertainty: f_x(a_x) = 1 - d_x and f_x(1 - a_x) = d_x. The flip
    acts independently on each homologue, including repeated pairs (x, x). This is not exact integration over
    one shared latent allele; a certain partner can still set the orientation.
- **Panel-allele uncertainty, leave-one-out** (3.4, `family_phase/allele_evidence.py`). A label's panel allele was
  made from the reads of the samples painted on it. Where few samples carry a label, a sample's own reads are a large
  part of that evidence, and a model that took the allele as known would count them twice: once inside the allele,
  once as the sample's own evidence. Up to 3.3 this made the genotype scores of samples on weakly carried rows
  over-confident (about 18 times nominal at scores of at least 0.9999 on K30, 4 times on std).
  - d_x is a capped empirical-Bayes per-copy uncertainty, conditional on the fitted carriers and panel, with the
    sample's direct own likelihood term removed. Carriers are samples whose solved MAP pair holds x and whose
    partner has a known allele; founder-QC-excluded samples are not carriers. This is not a cross-fitted refit or
    a posterior integrating over carrier/panel selection. Same-data selection dependence remains.
  - Each carrier adds e = l(g1) - l(g0), the refit's robust genotype log likelihood (robustness 0.01) of the allele
    1 against allele 0 given its other label's allele (2x for a carrier of x on both homologues). The released
    sample's own term is removed from the labels of its own MAP pair.
  - The empirical prior P(A = 1) is the share of allele 1 among the chromosome's called refit row-by-site cells. It
    reads no truth, label or metadata; it is estimated in each run.
  - d_x = sigmoid(logit prior + evidence) for a panel allele 0, one minus it for an allele 1, at most 1/2 (the refit
    decides the alleles; the cap therefore depends on the selected allele), and 0 below 1e-15.
  - Where every d is 0 (well-carried labels) the site model is the 3.3 model exactly.
- The sample's own genotype likelihood at the site weighs these alternatives, normalised per label pair. Reads that
  contradict the labels' genotype lower, withdraw or change the call; a site without reads keeps the labels'
  genotype at a score of at most 1 - 2/3 `site_deviation` (lower where a label's allele is uncertain).
- This is a modular conditional release, not the posterior of one generative model: the label-pair posterior comes
  from the painting's emissions (pre-refit panel, robustness 0.01, tempered by the design effect), while the site
  model uses the refit alleles and its own mixture. Direct own-term subtraction leaves indirect dependence through
  carrier/panel selection; the two recipes are not reconciled. `site_deviation` = 1e-3 was chosen on the validation simulations (0.01 unphased
  12-13% of the phased joins; 3e-4 kept more genotype errors).

- **Genotypes** come from the unanchored posterior, since the homologue order does not change unordered genotypes.
  A genotype is called where its posterior reaches `genotype_posterior` (0.9); otherwise both alleles are unknown.
  Family phase may therefore call, change or withdraw genotypes relative to the painting.
- **Homologue order.**
  - A sample with a released parent is ordered by parental origin: homologue k is the slot assigned to the slot-k
    parent (its copy where the chain's source is 0 or 1, not in not-a-copy stretches).
  - A sample with children and no released parent is ordered by its children's chains.
  - A gauge sample (neither) is anchored at its first confidently heterozygous bin, where P(heterozygous label pair)
    reaches `phase_posterior`, by the convention that homologue 0 carries the smaller label. The convention fixes
    the gauge, never a decoded label.
- **Phase sets.**
  - A gauge sample's set ends at the first heterozygous bin where, given the anchor, the orientation of its alleles
    is uncertain, or at a forced break, which is recorded at its own bin. The orientation certainty of a bin is the
    median, over its sites with P(heterozygous) ≥ `genotype_posterior`, of max(P01, P10) / (P01 + P10) under the
    anchored posterior; the set ends where it falls below `set_posterior` (0.999). This is stricter than the
    per-site `phase_posterior` (0.99) because a wrong continuation misphases the rest of the set, whereas a wrong
    site call misphases one site (at 0.99, near-ties decided by the label-switch prior alone added released long
    switches in partial-genome designs). It is a working value, not a calibrated confidence for a whole set.
  - Longer sets carry an upstream identity error over more sites: where two founder rows exchange identity at a
    point the assembly does not flag as a junction candidate, a sample carrying both rows keeps its unordered
    labels, so its own reads cannot see the exchange. A new partition can also withdraw a site that an earlier
    partition would have phased. The next set starts there and is
    anchored at its first confidently heterozygous bin; a partition without one releases no phase.
  - Uncertainty about which labels a sample carries is not phase uncertainty: where the candidate label pairs carry
    the same alleles at the sample's sites, the orientation is determined and the set continues. (Until 3.1 a set
    ended wherever the posterior of the decoded ordered label pair fell below `phase_posterior`. On real data most
    gauge-sample sets started at such label-uncertainty stops, although their allele orientation was still certain.)
  - Other samples' sets break only at forced breaks (for a sample with children and no released parent, the tie
    breaks it keeps; above).
  - Component edges always start a new set.
- **Phased calls.** A heterozygous call is released phased where its ordered posterior reaches `phase_posterior`
  (0.99). Otherwise it is released unphased: `frame_undetermined` bits 0 and 1.
- **Agreement of the two conventions.** Each sample is released from both solves, and only their agreement is
  released:
  - a genotype is called where both call the same genotype, otherwise both alleles are unknown;
  - a heterozygote is phased where both phase it;
  - phase sets break at both solves' breaks and wherever the two solves' relative orientation changes. Within a set
    they then agree up to one global flip, and the default convention's orientation is released.
- **Meiosis chains.** For every released parent edge, the child's unanchored block posterior (the same
  forward-backward as its genotypes) gives per bin the probability that the linked homologue copies the parent's
  homologue 0 or 1. Within each stretch between resets, the mirrored solve's homologue numbers are aligned to the
  default solve's by the parent's labels (a stretch whose two label orders disagree releases nothing). A bin is
  released (`chain_source`) where both solves give the same parental homologue at least `phase_posterior`, with the
  smaller posterior as its support. The chains reset at each component's first bin, at meiosis resets (protected
  intervals) and at the parent's and the child's released phase-set starts; a start inside a bin blocks that bin.
  [Recombination](recombination.md) calls a crossover between consecutive released bins of different homologue.
- **Interpretation.** Calls and phase are thresholded on normalised conditional scores of a working model with a
  fixed panel, a fixed pedigree and tempered emissions. Each block is evaluated around the final joint
  configuration. Uncertainty in alternative neighbouring configurations, pedigree hypotheses and shared panel
  identity is not integrated.
  - The thresholds are decision rules. They are not a claim of 90% genotype accuracy or 99% phase-join accuracy on
    arbitrary data, and their error behaviour is measured on simulations by design and event type.
  - The phase test reads one site's ordered allele marginal. That is not the probability of a join between two sites
    or of a whole set.
  - Phase sets have independent orientations. An unphased call and a phase break mean different things.
  - Slot assignment, relative phase and parental origin are distinct claims.
  - Junction candidates and withheld joins are diagnostics, not probabilities of a panel exchange.

## Product and records

`FamilyPhaseProduct` (`family_phase/records.py`, record `family_phase` per contig, schema 4):

| field | type | meaning |
| --- | --- | --- |
| `sample_ids` | N | run sample order |
| `positions` | `int64[L]` | every evidence site of the contig |
| `component_ids` | `int32[L]` | assembly component, −1 between components |
| `bin_edges`, `bin_edge_offsets` | `int64[M]`, `int64[C + 1]` | painting-bin edges (bp, half-open bins) per component; recombination reads its bins here |
| `bin_centers` | `float64[M − C]` | each painting bin's centre, the mean position of its sites, which the meiosis transitions use (components concatenated) |
| `calls` | `int8[N, L, 2]` | released alleles (−1 unknown, 0 REF, 1 ALT) in released homologue order (homologue k of a sample with a released slot-k parent is the slot assigned to that parent) |
| `displayed_frame`, `decoder_frame` | `int8[N, L]` | zero (kept for readers of the 2.x layout) |
| `coverage`, `coverage_offsets` | `int64[R, 2]`, `int64[N + 1]` | each sample's painted site ranges [start, stop) |
| `edges_identity` | str | identity of the `ParentEdges` the phase is conditioned on |
| `diagnostics` | `FamilyPhaseDiagnostics` | per component: sweeps, settled, J after the initialisation and every sweep, junction candidates; per chromosome: withheld joins, tie breaks, called alleles, phased and unphased heterozygous calls, phase sets, elapsed seconds |
| `frame_undetermined` | `int8[N, L]` | bits 0 and 1 set on a called heterozygote whose phase is not released |
| `phase_breaks`, `phase_break_offsets` | `int64[B]`, `int64[N + 1]` | per sample, marker boundaries i (between sites i and i + 1) where a new phase set starts; component edges are not listed |
| `chain_source` | `int8[E, M − C]` | per parent edge (`ParentEdges` order) and painting bin (components concatenated): the released parental homologue (0 or 1, in the parent's released homologue order) the child's linked homologue copies, −1 where none is released |
| `chain_support` | `float32[E, M − C]` | the smaller of the two solves' posteriors of the released homologue (0 where none) |
| `chain_reset` | `bool[E, M − C]` | the chain is not followed from the previous bin: a component's first bin, a meiosis reset or a released phase-set start of the parent or the child |

Records before the meiosis chains (schema below 4) are not read: rerun the stage. Evaluation and export read `sample_ids`,
`positions`, `component_ids`, `calls` and the released ambiguity fields. Export starts phase sets at component edges
and explicit phase breaks. Unphased calls are written unphased without splitting the set, and evaluation scores those
same sets ([simulation](../simulation.md#phase-pedigree-and-map-metrics)). `OUTPUT/family_phase/summary.csv` has one row per contig: `samples`, `sites`, `components`, `sweeps`,
`settled`, `objective` (final J), `junction_candidates`, `withheld_joins`, `tie_breaks`, `called_alleles`,
`phased_heterozygotes`, `unphased_heterozygotes`, `phase_sets` and `elapsed_seconds`.

## Implementation and cost

- **Structure.**
  - Each chromosome is one task of a chromosome executor sharing the stage's CPU budget, which bounds how many run
    at once. Each reports a byte estimate to the executor (four times the dense emissions of its largest component,
    including the release pool's shared copy, plus 48 bytes per sample and site for the release's per-site arrays
    and the float32 genotype likelihoods with a component's slice of them, plus 2 GiB); the memory governor
    paces new work by measured memory pressure, and the estimates do not lower the worker count.
  - Within a component, the two ascents (one per tie convention) run in the calling process; their kernels are
    serial Numba.
  - The releases are independent given the final state. They run in an `execution.BudgetedPool` whose workers
    share the final arrays, with one sample per task. Each task reports a per-sample estimate (block tables,
    forward and backward passes, posteriors, per-site projections). The pool's dispatch follows measured memory
    pressure; the estimate does not bound the number of workers.
- **Cost.** With K founders, S ≤ K + 1:
  - a block has at most 2(S + 2)² states and costs O(B S³) per sweep, plus O(B S² A_n) for its A_n descendants;
  - a block's or pairing move's gain is evaluated over the samples it can change, the sample and its A_n
    descendants, O(B (1 + A_n)), and J once per sweep, O(N B); a sweep is O(B (N S³ + S² Σ_n A_n)), linear in N
    when ancestries are bounded (Σ_n A_n is the number of ancestor-descendant pairs, at most N times the number of
    ancestors per sample). Before 3.4 every block was checked against the full objective, O(N B) per block and
    O(N² B) per sweep: ~3 h per chromosome at N = 2,500;
  - the release costs one forward-backward per sample, plus one per new anchor of a gauge sample, and a projection
    of O(m S²) for its m sites (a gauge sample's partition also projects the sites from its anchor to its next forced
    break, to find where its orientation becomes uncertain; a sample with children reads its tie gate from its own
    projection);
  - both the ascent and the release run once per tie convention: about 1.6 times the cost of one.

  The ascent settles in one to three sweeps on the validation designs.
- **Memory.**
  - Dense pair emissions take float64 N × S² × B: about 150 MB for N = 100, S = 8, B = 3,000, but 230 GB at
    N = 10,000, S = 31. Large inputs will need streamed or sparse emissions.
  - A release holds its block's tables (B × states²), per-bin label-pair posteriors (B × S²) and its sample's site
    likelihoods (m × 3). It never holds a per-site pair tensor.
  - The stage holds the chromosome's evidence likelihoods as float32 N × L × 3 for the site model: about 4 GB at
    N = 1,000 and L = 330,000.
- **Panel-allele uncertainty.** Per component and solve, the carriers' evidence costs O(N m) (in site chunks), and
  each release adds its own pair's terms, O(m). The site posterior stays O(m S^2) additions; only pairs of two
  uncertain labels are normalised one by one. On the validation designs family phase took 1.07-1.28 times as long
  (paired timings, 8 CPUs).
- **Resume.** `identity/family_phase` is bound per contig before any work and checked on resume. A contig with a
  `family_phase` record is not recomputed; an interrupted contig is recomputed.
- **Determinism.** Block order, ties and releases are fixed, and the pool returns the releases in sample order, so
  results do not depend on the thread count.

## Interpretation and limitations

- **Conditional uncertainty.** A parent's released phase fixes its children's chains at the final state, and a
  child's fixes its parents'. The exact joint can be far less certain than these conditionals (a four-bin, two-sample
  counterexample gives 0.5 against 0.999999). The withheld joins and tie breaks cover the cases measured on the
  validation designs; they are not a general calibration.
- **Panel identity.** An exchange that no single carrier crosses is not a junction candidate: carriers of both rows
  only, or chimeric and duplicated rows. A double carrier's phase across it can then be confidently wrong (8c chr2:
  one row follows two founders). Supplying this uncertainty is an upstream contract (assembly and founder
  refinement).
- **Samples without released parents** are phased as their own mosaics, plus withheld joins and ties. Shared latent
  parents (sibships) are not modelled. Most samples of real crosses whose parents were not sequenced fall here.
- **Panel versions.** Emissions use the painting's panel; released alleles use the refit panel. They differ at a few
  hundred named-allele cells per component.
- **Measured score behaviour** (validation simulations; site-level events: a called genotype, and the relative
  orientation of two consecutive phased true heterozygotes of one set). Genotype and ordered-allele scores combine
  conditional label-pair distributions from two fitted family states with the site model and the sample's genotype
  likelihoods; the smaller of the two solves' scores is reported on their agreed release. They are working
  conditional release scores, not calibrated probabilities: a threshold selects releases without guaranteeing its
  error rate, a per-site 0.99 phase threshold bounds a two-site join only at 0.98, and shared panel errors, read-model
  errors and the uncertainty of conditioning relatives are not represented.
  - Without the site model, genotype scores were overconfident in every top bucket (std, scores at least 0.9999:
    22,885 wrong against a nominal 52). With the 3.3 site model they were conservative on most designs but
    overconfident where rows have few carriers: at scores of at least 0.9999, 18 times nominal for K30 samples
    without a released parent and 4 times for std's. With the leave-one-out panel-allele uncertainty (3.4) no
    genotype bucket at or above 0.999 exceeds twice nominal on any of the 13 validation designs (K30 0.24 times, std
    0.30 times); genotype errors fell on 12 of them (std -37%, K30 -51%). The cost is phase where a sample's
    heterozygote rests on weakly carried rows: K30 phases 2.5 points fewer correctly called true heterozygotes, std
    0.4 and seed 0.2.
  - Gate 1 measured min(p, 1-p) on mapped single-row labels, while release uses capped disagreement with the
    selected allele. Pooled and unmapped labels are unvalidated. Few-carrier classes were conservative; correlated
    carrier identity errors remain a failure mode, with 3-9 times overconfidence in four designs' well-carried
    classes. These observations do not establish calibration for the selected-allele-dependent capped quantity.
  - A heterozygous call where both labels carry the same allele is phased by the label whose allele is more likely
    to be wrong. Where the other label's homologue departed instead (a conversion or a label error), the site is
    misphased. The independent main totals show 2-10 added switch events on four designs while the total falls
    from 693 to 495.
    Across the 13 saved CAL2 replays, within-released-set neighbour scoring identified 29 misphased sites
    among 2,005 assessable gains out of 2,028 newly phased heterozygotes (21 lacked a neighbour and two had
    tied frames; f3 remained 5 of 22), and 3,886 among 165,229 assessable losses out of 165,241 heterozygotes
    phased by K but released unphased by CAL2 (11 lacked a neighbour and one had a tied frame).
    This auxiliary diagnostic does not establish that each misphased gained site causes an added switch event;
    the main genotype/switch totals are unchanged by the recount. TroMau gained/lost counts measure coverage and
    consistency only, because individual truth is unavailable.
  - Adjacent phase joins are conservative except on K30. There, single-site phase errors of the founder panel (two
    rows carrying each other's alleles at one site, already in the painting's panel) make confidently wrong joins
    that no sample's reads can see: 180 wrong against a nominal 132 in the 0.999-0.9999 bucket.
- **Resolution.** Inheritance switches are placed at bin resolution.

### Site storage and memory

Each chromosome uses resident storage while its decoded likelihood array is at most **8 GiB**; larger arrays
use site windows and temporary mapped outputs. The footprint is computed from the record's exact shape and dtype
(24*N*L bytes for float64 genotype likelihoods). It is an evidence-buffer budget, not a fitted peak-RSS estimate,
a worker-admission rule or a scientific threshold. All eight measured N=1,003 chromosomes fit (largest 7.385 GiB).
At the same chr1 site count, N=2,503 needs 18.43 GiB of evidence and selects windows. Short chromosomes can remain
resident at larger N: the 10%-site n2500c chr1 crop has 1.843 GiB and remains resident automatically.

Resident storage retains dense likelihood reads, whole-sample site projections and the original per-solve release
scheduling. Windowed storage uses the bounded buffers described below. Both modes use the same B/D2/FPD solver,
CAL2 evidence and leave-one-out arithmetic, priors and decisions. The stage API's `storage="resident"` and
`storage="windowed"` overrides support exact replay comparisons; production uses `storage="auto"`. Storage mode
is not part of the scientific configuration or record schema. Windowing trades additional I/O and site-loop work
for memory; resident mode avoids that cost when evidence fits the budget.

In windowed mode, the coordinate ascent, D2 parent-label moves and both tie conventions still see each complete component's bin
state. Site evidence is read from checked checkpoint chunks for one sample at a time, in windows of 4096 sites
extended to the end of the last bin. Site projections and orientation medians use these same whole-bin windows;
phase-set construction and the agreement between conventions retain their complete sample context. No likelihood,
reduction order, threshold or scientific decision changes.

Completed samples are consumed immediately. The chromosome's calls, ambiguity, legacy zero frames, optional scores
and final chain arrays are ordinary ndarrays backed by temporary files. Initialisation and checkpoint compression
use bounded byte chunks; completed contiguous output pages are flushed and removed from process mappings, with clean file-cache eviction advised. The files are unlinked while
their mappings remain owned by the arrays, and disappear when the arrays are released. The record schema and values
are unchanged. Temporary disk space is required (5 bytes per sample/site for the mandatory site arrays, another 8
when diagnostic scores are requested, plus chain arrays). These files and the CAL2 spool below use the checkpoint
store's root on the run's own storage, independently of TMPDIR or Python's temporary directory. They are anonymous
and unlinked while open; checkpoint scans and resume never see or depend on them, and no name survives a crash.
V4 changes only this location; four forced-windowed replays give bitwise-identical results to v3.

Provision disk space on the checkpoint filesystem for the chromosomes that family phase runs at once. If that
filesystem is memory-backed, the file backing still consumes memory. Together with the CAL2 component spool
(12 N m bytes for m component sites),
the mandatory site arrays (5 N L bytes) and optional scores (another 8 N L bytes) sum conservatively to 13.05 GiB,
or 19.20 GiB with scores, at N = 2,503 and L = m = 329,391, before chains and checkpoint-writing space. This is
logical-array accounting, not a measured simultaneous peak or allocated disk blocks: lifetimes differ and
zero-filled files can be sparse. File truncation does not reserve the disk blocks in advance. Process RSS/PSS
excludes some filesystem cache, and eviction advice does not impose a hard job-memory bound.

The memory saving relies on whole mapped products not being pickled between workers: unpickling can create a
resident copy. Production never pickles these products between workers: the chromosome worker writes its product
locally and returns a summary, and release tasks transfer bounded sample rows and results.

The working memory bound applies to the site tensors, not to the whole inference model. Full bin emissions,
forward/backward posteriors, pedigree label states, painting/refit inputs and phase boundaries remain resident.
One checked compressed-chunk cache serves the coordinating process. It gathers at most twice the worker count
of float32 sample likelihood rows (12 bytes per released site per row) for each release batch. Workers retain
one sample's release arrays and project sites through windows; compressed caches are not duplicated in workers. A bin larger
than the target window is processed whole to preserve its median. Consumers requesting dense copies of the final
arrays can still allocate chromosome-sized memory; the family-phase writer streams them.
CAL2's carrier evidence uses its original 4096-site chunks, reading only that chunk's genotype likelihoods
for all samples from a temporary float32 component spool (12 bytes per sample and released site). The spool is
filled one sample/window at a time so the compressed evidence is not repeatedly decompressed per site block; its
resident pages are discarded between carrier chunks. Each site's homologue-0 and homologue-1 additions keep their original order. Both solves' S x sites
evidence and flip matrices remain resident, as do each worker's current sample leave-one-out terms. The original
CAL2 flip-probability and own-term functions keep their full input shapes and arithmetic. Each solve receives its
own uncertainty matrices; neither the panel prior nor any flip threshold changes.
