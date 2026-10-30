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
- **Painting ties.** A sample without a released parent also breaks its sets at every painting tie boundary. These
  are the released chunk boundaries where the painting's track order was set by a tie: the ordered and the swapped
  continuation change equally many candidate rows and allele-visible keys. A linked homologue is ordered by parental
  origin, so painting ties do not affect it.
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
- Given a label pair with known alleles, the sample carries those alleles with probability 1 - `site_deviation`
  (1e-3), or else departs from them with a uniform genotype: a panel allele error, a label error shorter than a bin,
  a conversion or a mutation.
- The sample's own genotype likelihood at the site weighs these alternatives, normalised per label pair. Reads that
  contradict the labels' genotype lower, withdraw or change the call; a site without reads keeps the labels'
  genotype at a score of at most 1 - 2/3 `site_deviation`.
- This is a modular conditional release, not the posterior of one generative model: the label-pair posterior comes
  from the painting's emissions (pre-refit panel, robustness 0.01, tempered by the design effect), while the site
  model uses the refit alleles and its own mixture. Within a label pair the site's reads enter once, but the two
  recipes are not reconciled. `site_deviation` = 1e-3 was chosen on the validation simulations (0.01 unphased
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
  - A gauge sample's set ends at the first heterozygous bin whose ordered posterior, given the anchor, falls below
    `phase_posterior`, or at a forced break, which is recorded at its own bin. The next set starts there and is
    anchored at its first confidently heterozygous bin; a partition without one releases no phase.
  - Other samples' sets break only at forced breaks.
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
same sets. `OUTPUT/family_phase/summary.csv` has one row per contig: `samples`, `sites`, `components`, `sweeps`,
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
  - every applied or rejected block is checked against the full objective, O(N B) per block, so a sweep is O(N² B)
    without the block terms;
  - the release costs one forward-backward per sample, plus one per new anchor of a gauge sample, and a projection
    of O(m S²) for its m sites;
  - both the ascent and the release run once per tie convention: about 1.6 times the cost of one.

  The ascent settles in one to three sweeps on the validation designs.
- **Memory.**
  - Dense pair emissions take float64 N × S² × B: about 150 MB for N = 100, S = 8, B = 3,000, but 230 GB at
    N = 10,000, S = 31. Large inputs will need streamed or sparse emissions.
  - A release holds its block's tables (B × states²), per-bin label-pair posteriors (B × S²) and its sample's site
    likelihoods (m × 3). It never holds a per-site pair tensor.
  - The stage holds the chromosome's evidence likelihoods as float32 N × L × 3 for the site model: about 4 GB at
    N = 1,000 and L = 330,000.
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
    22,885 wrong against a nominal 52). With it they are conservative on four designs and within about twice
    nominal on std, but still overconfident on K30 (1,064 wrong against 61).
  - Adjacent phase joins are conservative except on K30. There, single-site phase errors of the founder panel (two
    rows carrying each other's alleles at one site, already in the painting's panel) make confidently wrong joins
    that no sample's reads can see: 180 wrong against a nominal 132 in the 0.999-0.9999 bucket.
- **Resolution.** Inheritance switches are placed at bin resolution.
