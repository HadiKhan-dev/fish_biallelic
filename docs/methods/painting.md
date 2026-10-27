# Painting

Painting decodes every sample, on each assembled founder component of a chromosome, as a diploid mosaic of the
component's founder rows plus an explicit BACKGROUND state for haplotypes the panel does not represent, each track
copying its founder faithfully or through a run where the panel row is locally wrong, and releases per sample, track and
bin the most specific ancestry statement whose within-model posterior reaches a fixed threshold. The stage (`painting`,
entry point `painting.run_contig`) runs per contig in the contig's lane, right after the final assembly. Per component
it first estimates the copying model from all samples ([copying model](#copying-model)); label continuity then relabels
row suffixes of the assembled panel where two rows exchange founder identity at a link point; the relabelled panel is
the one painted, published and read downstream. Pedigree inference, the [founder refit](founder_refit.md) and [family
phase](family_phase.md) read the painting; [recombination](recombination.md) does not (its bins come from the
family-phase product). Nothing downstream feeds back into the painting or the assembly.

## Inputs and outputs

| | record or setting | content |
| --- | --- | --- |
| reads | `assembly/final` | the final assembly's components as `ComponentPanel`s: positions, released calls (with completion fills), inference calls (the frozen pre-fill snapshot), kept sites, atomic spans, break reasons, count call |
| reads | `evidence` | normalised genotype likelihoods `float64[N, L, 3]` and the positive-depth mask `observed bool[N, L]` of **all** samples, including those the sample-quality QC keeps out of founder construction |
| reads | the contig's genetic map | `[genetic_map] file` and `[genetic_map] rate_cm_per_mb` (fallback rate); see [genetic maps](../genetic_maps.md) |
| writes | `identity/painting` | `PaintingConfig`, the identities of `assembly/final` and `evidence`, the map's digest, code identity |
| writes | `painting/summary` | per-component shapes (founders K, classes C, states S, selected sites, bins, packed emissions kept), eligible sample-component pairs, label-continuity totals |
| writes | `painting` | the `ChromosomePainting` product; its presence marks the contig complete |
| publishes | `OUTPUT/painting/<contig>/` | eight diagnostic tables ([published reports](#published-reports)) |

`PaintingConfig` has no user-visible field; its constants are internal model constants listed in
[the configuration reference](../configuration.md).

## Label continuity

Each founder row of an assembled component is a path through the local rows of consecutive atomic blocks (the
prepared panels). Next to stretches where rows are locally interchangeable (identical copies inside a founder's
run of homozygosity, haplotypes shared between founders), linking and refinement can let two rows exchange founder
identity at a link point, the boundary between two atomic blocks. The alleles at every site remain right as a set,
but painting reads label continuity as haplotype continuity: every sample carrying either haplotype across the
point pays a track switch there, or keeps its labels by exchanging its two tracks, which flips a parent's phase
frame and puts a false crossover into every child.

Label continuity (`painting/label_continuity.py`) repairs such exchanges with the painting model itself, the
component's copying model estimated before the search and fixed throughout it. A suffix transposition t exchanges the
labels of rows a and b at every site from link point j onwards. The set of row alleles at every site is unchanged and
the painting state space is renamed, not changed (the panel-error states of a and b move with them), so every
emission is unchanged except in the bin that straddles j. Candidates are screened with the label-level copying process
(the estimated rate on the tempered emissions, without the panel-error state, whose per-bin mixture is not a sum over
a straddling bin's parts): for every sample s,

    V_s(t) = max_x [ G_s(x) + L_s(x) + Q_s(t(x)) ],   Q_s = P_s + B_s,

where G_s is the max-product forward message into the bin holding j, L_s and P_s are that bin's emissions on its
sites left and right of j, and B_s is the backward message from that bin. The Hamming transition is invariant when
both of its arguments are renamed, so V_s(t) is the exact best-path score of the relabelled panel under that process:
paths re-optimise on both sides, and a switch anywhere inside a stretch of identical rows vanishes when the
relabelling makes the haplotype continuous. The screen only ranks and admits candidates; every decision is made on
the painted switches of the full model (below).

Search and acceptance use no pedigree, cohort label or truth:

- Link points are atomic-span boundaries strictly inside the component's selected sites. Only singleton anchored
  trajectory classes move; pooled duplicate rows and BACKGROUND keep their labels. A component with fewer than two
  movable rows, no link point, fewer than two bins or no eligible sample is painted unchanged.
- Candidates (link, row pair) are ranked by their summed score gain over the eligible samples and considered only
  when the gain reaches `PaintingConfig.label_continuity_minimum_switch_gain = 2.0` switch units u, where u is the
  median over the component's bin edges of the log-probability cost of one track change against staying,
  log(1 − θ) − log θ + log(S − 1) ([copying model](#copying-model); `PaintingTransitions.switch_unit`); one painted
  switch costs about u.
- A candidate is decided on painted switches exactly as painting counts them (label changes of the fewer of the two
  traceback conventions' paths; release 2.14, before which label continuity counted the run's convention alone):
  every eligible sample is re-decoded with the painting Viterbi (the class x fidelity chain) under the candidate labels
  (release 2.14; see below). It
  is accepted when the total falls by at least 2 switches and the samples whose decoded count rises (losers) number
  at most `PaintingConfig.label_continuity_maximum_loser_ratio = 0.1` times those whose count falls (winners). The
  intent: under a label exchange, decoded carriers that span the link without a real change of haplotype gain no
  switch; under a real crossover inherited by one lineage, decoded carriers of the two haplotypes from other lineages
  that span the link would gain one, so the ratio rule keeps such a crossover when those carriers are painted. Who is
  counted depends on read coverage and on the decoded paths, and where no other carrier spans the link the rule
  cannot tell the two cases apart (Interpretation and limitations). Score changes do not classify samples.
- The first acceptable candidate in ranked order is applied, every message is recomputed, and the search repeats
  until no candidate is acceptable. Each accepted move lowers the painted switch total, so the search terminates.
- Competing repairs (release 2.14), a stated policy. When another acceptable candidate of the accepted one's tie
  class of gains (`maths.ties.tie_classes`, connected classes) would relabel differently (another row pair, or the
  same pair at a non-equivalent link point), neither is applied in that iteration: the search continues with the
  next class, and the event is counted (`painting.relabel_ambiguous` in the tie report). A later class or iteration
  may still move the same rows. Before 2.14 the first in index order was taken, so the input row order or the
  convention reversal could choose a different founder panel (a four-row fixture: 450 cells apart after the best
  row matching; now no move under every row order, convention and sample order).
- Tied suffix transpositions (release 2.14). Where the two rows carry identical founder-axis data (released and
  inference calls, ALT probabilities, support) at every panel site between two consecutive link points, transposing
  at either point gives the same per-site panel data, painting and switch counts (the stored routing of rows over the
  prepared panels may differ), and the placements are checked once. An accepted move is placed at the leftmost link
  point of such a run, a stated rule that the convention reversal does not change (before 2.14 the reversal moved it
  to the rightmost one; the painted calls were already identical).
- Error capabilities (release 2.14). A candidate that would change which of its two classes is called at some
  selected site (and so which classes have a panel-error state) is not accepted: the model fixed before the search
  does not describe the other alphabet (counted as `painting.relabel_capability_change`).
- The relabelled component is kept only if repainting it with the painting functions lowers the total and leaves
  its trajectory classes, selected sites and error capabilities unchanged.

Every eligible sample is re-decoded for each distinct candidate (release 2.14). Before, a sample whose best state at
the link strictly preferred a state involving neither transposed row was skipped: a relabelling maps the optimal paths
one to one, but it can change which of several exactly co-optimal paths a traceback convention selects, and so the
selected switch count (a review's finite HMM counterexample), so the shortcut was not a certificate. Each re-decoded
sample is bitwise what a full decode gives, and a check keeps only the per-sample count changes.

Per component the outcome is a `LabelContinuity` record: `reason` (`relabelled`, `no_link_or_movable_row`,
`no_evidence`, `no_move`, `trajectory_classes_changed`, `error_capabilities_changed` or `painted_switches_not_lowered`),
link points, movable rows,
u, switch totals before and after, candidate checks, the accepted moves (atomic span, bp position, rows, score gain
in switch units, painted switch change, winners, losers) and, when accepted, the row maps (in atomic span a, painted
row k carries assembled row `row_maps[a, k]`). Released and inference calls are permuted together; the painting
record stores the painted panel as row paths over the assembly's prepared panels, while the assembly records keep the
unrelabelled rows. Label continuity scores all samples, as the painting does. The log reports each contig as
`[label continuity] <contig>: m row-suffix relabellings (k identity-ambiguous); painted switches a -> b`.

Founder identity across a move (release 2.14, a diagnostic). A label exchange of two founders' rows makes the
carriers of either founder switch at the link, in both directions; one shared recombinant lineage switches one way
only. Each accepted move records its winners' directions: `winners_forward` (a track in row a's state left of the
link and b's right of it, within two bins, on the default convention's decoded path before the move) and
`winners_reverse` (b to a); the move is `identified` when both are positive. A one-way move is kept, with its
founder identity across the link reported as ambiguous (`label_continuity.tsv`); no consumer reads the flag, so it
changes no released label.

## States and bins

The state space of a component is built from its inference calls (int8, -1/0/1, the frozen pre-fill snapshot of
assembly), not from the released calls with completion fills (`painting/states.py`).

- **Selected sites.** The component's kept (discovery) sites at which at least one row is called. A site without
  any founder call carries no ancestry evidence.
- **Trajectory classes.** Rows whose calls agree at every selected site cannot be told apart by any sample and form
  one class (C classes, ordered by their first row). A class called at no selected site is unanchored.
- **Track states.** The C classes plus BACKGROUND, S = C + 1 states per track; a sample's state is an ordered pair
  (S^2 states). BACKGROUND's ALT probability at a site is the Beta(1, 1) posterior mean of the class calls there,
  (1 + ALT calls) / (2 + calls); a class uncalled at a site takes the same probability there.
- **Bins.** The Ls selected sites are split into B = max(1, floor(Ls / n)) near-equal consecutive runs, with
  n = `PaintingConfig.snps_per_bin = 100` (the first Ls mod B bins hold one extra site). A bin's centre is the mean bp
  of its sites; bin b owns the bp interval [edges[b], edges[b + 1]), with inner edges midway (rounded up) between
  the last site of one bin and the first site of the next, edges[0] the first selected site and edges[B] the last
  plus one.
- **Local allele-equivalence keys** (`local_keys int32[B, S]`). In each bin, anchored classes that are called at one
  of the bin's sites and have identical -1/0/1 vectors over the bin share the key of their smallest class (exact
  partition refinement, O(C) per site); other anchored classes key to themselves; unanchored classes and BACKGROUND
  key to the public UNKNOWN index C. Two classes are indistinguishable by reads in a bin exactly when they share a
  key below C.

## Emissions

The emission of a sample, an ordered state pair (x1, x2) and a bin sums, over the bin's observed selected sites, the
robust genotype log likelihood

    e = max(log[(1 − ε) Σ_g GL(g) P(g | x1, x2) + ε/3], f) − max(log(1/3), f),

with ε = `PaintingConfig.robustness_epsilon = 0.01` and the floor f = `PaintingConfig.emission_log_floor = -50.0`.
GL rows are the evidence stage's normalised likelihoods (exactly 1/3 each at zero depth); unobserved sites are
skipped and the centring by log(1/3) makes flat sites neutral. P(g | x1, x2) uses each track's ALT probability q
(its class's call, else BACKGROUND's frequency) with independent homologues, P(0) = (1 − q1)(1 − q2) and
P(2) = q1 q2, except that two tracks in the same class share one latent allele (P(1) = 0, P(2) = q). Emissions are
symmetric in track order, so each unordered pair is evaluated once.

**Design effect (release 2.14).** Each sample's bin log emission is then divided by 1 + (n − 1) r, n the sample's
informative sites in the bin (observed sites whose genotype-likelihood row is not flat; a bin without any has weight
1) and r = `PaintingConfig.emission_site_correlation = 0.1` (`painting.hmm.design_effect`). The bin sums its per-site
terms as if they were independent given the copying pair, but panel errors run along rows, and reads and
genotype-likelihood errors are correlated between nearby sites, so the sum overstates the evidence of a bin. The
divisor is the variance inflation of a sum of n equal-variance terms with common correlation r (Kish's design effect),
used as a tuned, equicorrelation-inspired magnitude adjustment of the composite log likelihood: under that working
assumption a full bin of 100 informative sites counts as about 9 variance-equivalent sites and a bin of 8 as about
4.7, and the weight tends to 1 as n → 1. It is not an estimated dependence correction: the composite-likelihood
literature calibrates such weights from sensitivity and variability information (Varin, Reid and Firth 2011,
Statistica Sinica 21:5-42; Pauli, Racugno and Ventura 2011, Statistica Sinica 21:149-164), while the genotype
contrasts here have unequal strengths and their dependence can vary with state and distance; r is a tuned constant
chosen on the simulations and sparse-read controls ([interpretation](#interpretation-and-limitations)). Every
consumer of the emissions uses them tempered: label continuity (each part of a bin divided by the whole bin's
weight, so the parts still sum to the bin), the stored packed emissions and pedigree's source chain.

## Copying model

Release 2.14 (`painting/transitions.py`, `painting/fidelity.py`). Each track (homologue) of a sample copies a
**label**, a trajectory class or BACKGROUND (S labels), with a **fidelity**: faithful, or in a panel-error run
(anchored classes only). A track state is (label, fidelity): S' = S + (anchored classes) ≤ 2S − 1 states per track.

- **Label process.** Along the genetic map the label changes as a jump process, independently on the two tracks:
  jumps at rate ρ per Morgan, each to one of the other S − 1 labels uniformly. Over a bin edge of d Morgans (between
  consecutive bin centres on the contig's map) a track ends on a different label with probability
  θ = (S − 1)/S · (1 − exp(−ρ d S/(S − 1))), at least `painting.transitions.MINIMUM_SWITCH_PROBABILITY = 1e-15`
  (θ ≈ ρd on short edges; across a long gap it tends to the uniform law instead of forcing a change). ρ is the
  component's effective ancestry-change rate, a pooled copying rate of its samples, not a number of meioses.
- **Panel-error chain.** A label that stays keeps or changes its fidelity by a two-state chain: faithful → error with
  probability a per edge, error → faithful with probability b (1/b bins: the mean length of a run that keeps its label);
  a label that moves draws its fidelity from the chain's stationary law (error with probability a/(a + b)); BACKGROUND
  and unanchored classes are always faithful. A faithful track emits its class's calls; an error track emits as
  BACKGROUND (alleles from the population frequencies): its row is locally wrong for this sample, or the sample's
  haplotype is locally absent from the panel, while the founder label continues. Two faithful tracks of one class share
  their latent allele, as in the emission above. A panel-error run thus contributes log(1/a) at its entry and log(1/b)
  at its exit (besides its duration and label-stay factors) and keeps the founder label, where a label-level model must
  pay two ancestry changes (an excursion) to escape a locally wrong row. Painted switches are label changes only.
- **Pair transition.** The product of the two tracks' transitions, so every recursion runs in two stages (track 1,
  then track 2), each O(S') per state from per-label sums: O(S'^2) per bin and sample.
- **Initial law.** At a component's first bin each track's label is uniform and an anchored label's fidelity follows
  the chain's stationary law (error with probability a/(a + b)), so the chain is stationary along the component.
- **Estimation.** (ρ, a, b) are estimated per component from all its eligible samples, on the assembled panel before
  label continuity, by EM: `PaintingConfig.estimation_iterations = 8` updates from the fixed starts `initial_switch_rate
  = 2.0`, `initial_error_entry = 1e-3` and `initial_error_exit = 0.25`. Each update is an exact scaled forward-backward
  pass over the chain (`painting.fidelity.chain_statistics`), then the M-step of the expected complete-data
  log-likelihood: ρ maximises Σ_e [n_e log θ_e + (m − n_e) log(1 − θ_e)] over the edges of positive length (n_e the
  expected label changes at edge e, m the eligible tracks; a 241-point log-spaced scan over [0.01, 1000] per Morgan
  refined by golden section, a numerical search since the objective can have more than one local maximum); (a, b)
  maximise s log a + k_F log(1 − a) + e log b + k_E log(1 − b) + i_E log a + i_F log b − (i_E + i_F) log(a + b) (s, e
  the expected error-run starts and ends, k_F, k_E the expected kept faithful and kept error transitions of labels that
  stay, i_E, i_F the expected anchored tracks drawn into error or faithful fidelity from the stationary law, at the
  first bin and after a label move; bounded quasi-Newton in log space). Neither update is accepted when it would lower
  its part of the objective, so the log-likelihood does not fall between updates. The estimates are clipped to a ∈
  [1e-6, 0.5], b ∈ [1e-3, 0.999] and rounded to `estimate_grid = 2^-6` in natural-log space (about 1.6%). The number of
  updates is fixed: a deterministic, truncated estimator, not a convergence test. a at its floor is a boundary fit:
  under this model, emission adjustment and schedule the data gave no support for error runs (it is not a test of their
  absence). On the simulations EM estimates runs (a about 1e-5 to 5e-5, 1/b about 10 to 30 bins) on seven of the eight
  std chromosomes, whose assembled rows carry error runs, and puts a at or near its floor elsewhere (std chr13, most f3,
  backcross and K30 components). 1/b is the mean length of a run that keeps its label (a run also ends with a label
  change). Per component the model is published in `painting_model.tsv`.
- **Label-level table.** `PaintingTransitions` (`log_same`, `log_one`, `log_two` per bin edge, with
  same + 2(S − 1) one + (S − 1)^2 two = 1: (1 − θ)^2, θ(1 − θ)/(S − 1) and θ^2/(S − 1)^2) is the label process alone,
  stored with ρ, a, b and r. Label continuity's screen and pedigree's source chain use it with the tempered
  emissions: reduced models without the panel-error state (not the marginal of the chain).

Before 2.14 a track change had weight θ' = θ exp(−100) (θ Haldane's fraction of the map distance, i.e. ρ = 1, and a
change of both tracks coupled), about 109 nats per change: a normalisable but imposed switch-suppression prior whose
effective change rate was not the map's and whose posteriors were near 0 or 1. 2.14 replaces it by the estimated
rate, the explicit panel-error runs and the emissions' design effect (the trade-offs measured are under
[interpretation](#interpretation-and-limitations)).

## Decoding

Every sample is decoded independently (`painting/fidelity.py`, `painting/hmm.py`):

- **Viterbi.** Exact max-product recursion over the S'^2 ordered track-state pairs, in two stages per bin (track 1, then
  track 2), each taking for every state the best same-label predecessor (its faithful or error state) and the best
  predecessor of another label (from the best and second-best scores by label): O(S'^2) per bin. Predecessors within
  `VITERBI_TIE` = 2^-15 nats of the best are tied (release 2.13: 16 times the gap motion under 1e-6 GL noise; since 2.14
  the scores are tempered, so the quantum sits further above that motion); per track the back-pointer then stays in the
  same state, else keeps the label, else takes the smallest index (a stated prior of fewest changes); the final state is
  the smallest index within the tie of the best. The dynamic-programming scores are exact maxima; each bin spends up to
  two tie allowances (one per stage), so the traced path lies within (2B − 1) x `VITERBI_TIE` of them. Scores are
  centred per bin. The mirrored convention decodes the same model backwards (transposed transition, the initial law at
  its last step). The released path is the label sequence of the Viterbi track states. The traceback runs right to left
  and stays while staying is tied, so among co-optimal placements a switch lands on the leftmost tied edge. That
  placement is a convention and is never released by itself: the release uses switch intervals (below). Under
  `HAPLOTYPES_TIE_REVERSE=1` (release 2.2 convention reversal) the recursion runs over the bins in reverse order and
  index ties take the largest state, so the switch lands on the rightmost tied edge.
- **Track order.** The two tracks of a decoded label path are a gauge. Where exchanging the tracks' label suffixes
  after a boundary changes equally many labels, the suffix is swapped when the swapped order has strictly fewer
  allele-visible label changes (a change between classes that share a local key below C in either adjacent bin is
  invisible), which keeps a label change between identical rows inside a pooled run off every bin. Under the
  label-level Hamming table such an exchange leaves the path score unchanged; under the class x fidelity chain it
  need not (the fidelity factors of the exchanged tracks differ), so the reordered labels are a representative of the
  decoded unordered label pairs, not necessarily a joint Viterbi path of the chain. Unordered pairs, posteriors and
  releases are unchanged by it. The released track order is then fixed from the combined candidate sets (below).
- **Release posteriors.** An exact scaled forward-backward pass over the chain gives, per bin, the posterior mass of
  every label pair, all and with both tracks faithful; qV = the mass of the unordered pair of public classes of the
  Viterbi labels (unanchored classes and BACKGROUND share the public UNKNOWN identity), qA = that of the unordered pair
  of local allele keys, and qF, qAF = the same masses with both tracks faithful: the probability that the pair's
  rows carry the sample's alleles in the bin. Keys are coarser than public classes and faithful copying is a subset,
  so qA ≥ qV ≥ qF and qA ≥ qAF ≥ qF. The release gate is qF and qAF (release 2.14).
- **Eligibility.** A sample is eligible on a component if it has an informative site there: a selected site that is
  observed and whose normalised GL row has a spread (maximum minus minimum) above 16 machine epsilons. Ineligible
  samples abstain in every bin.
- **Direct support.** A track is directly supported in a bin when its Viterbi class is called at an informative
  selected site of the bin.
- **Switch counts.** `switch_counts[s]` is the minimum number of unordered label changes along the Viterbi path (a
  pure exchange of the two tracks is gauge, not a switch, and a fidelity change is no ancestry change; the fewer of
  the two traceback conventions' paths, see below), 0 for ineligible samples.

### Switch intervals (release 2.2)

Where a track switches between classes a and b but the sample's reads cannot place the switch, the painting
releases the interval instead of a point (`painting/hmm.py` `switch_interval_masks`;
`painting/fidelity.py` `decode_candidates`):

- **Indistinguishable bins.** In a bin, a and b are indistinguishable for a sample whose other track is in state y
  when the pairs (a, y) and (b, y) have the same genotype law at every selected site of the bin that the sample
  observes (read depth > 0): the same call (0, 1 or uncalled; BACKGROUND counts as uncalled everywhere), and, where
  both are uncalled, not exactly one of them equal to y when y is a named class (two tracks in one named class
  share one latent allele and have no heterozygote; two different classes, or BACKGROUND with itself, do not; release
  2.14, before which every equal call, uncalled included, counted, so a switch the reads located could be widened
  into an interval). In such a bin the emissions of (a, y)
  and (b, y) are equal, so moving the switch across the bin changes the path score only through the transition
  prior (the map distance of the edge, or nothing on a cold stretch). The test reads only the observed mask and the
  calls, so genotype-likelihood noise cannot move it.
- **Interval of a switch.** For a track change a → b of the gauge-fixed Viterbi path after bin e, the interval is
  the maximal run of indistinguishable bins on both sides of e; it stops at the track's neighbouring switches, each
  switch keeping one bin of its own segments, so an interval never removes a switch. The bins of the interval get
  the candidate set {a, b} on that track; elsewhere the candidate set is the Viterbi state.
- **Both traceback conventions.** The Viterbi path of the mirrored convention is decoded too and its intervals
  taken. Per bin the two decodings' candidate sets are combined into the smallest unordered cover (the two track
  matchings give (P0 ∪ S0, P1 ∪ S1) and (P0 ∪ S1, P1 ∪ S0); the smaller is kept, and if both are equally small but
  differ, both tracks get every candidate). This also covers a double switch whose pairing of before and after
  classes on the two homologues is not determined.
- **Released track order (release 2.14).** The unordered pair of combined candidate sets of every bin does not depend
  on the traceback convention; the order of the two tracks read from one convention's path did (its final state
  and traceback run from opposite ends, so under the convention reversal about half of all sample-bins came out with
  the tracks exchanged). The track order (H1/H2, a gauge) is now a function of the unordered pairs alone
  (`canonical_track_order`): from the last bin, which puts the smaller candidate mask on track 0, leftwards, each
  bin takes the orientation with fewer track changes towards the bin after it (a track keeps its identity while
  consecutive candidate sets share a state), then fewer allele-visible changes, and when both orientations are
  equal, the smaller mask on track 0. The Viterbi states, intervals and direct support follow their tracks. Measured
  on the 2.13 final runs' panels (std 8 contigs, f3, backcross, K30, TroMau): the default and the reversed
  convention now release identical ordered products; against 2.13 the unordered products are identical and the
  track order differs in up to 5.1% of sample-bins (TroMau). The ordered arrays are per-bin representatives in that
  gauge, not necessarily one original ordered Viterbi path.
- **Release gate.** The gate is 2.1's rule on the faithful masses: qF and qAF are the posterior masses of the Viterbi
  labels' unordered public-class (local-key) pair with both tracks faithful, here the smaller of the two traceback
  conventions' pairs (the model's faithful-copy mass, not an empirical probability that the founder identity is right).
  The statements below hold for a fixed model and posteriors; the 2.14 model itself changes the posteriors. Where both
  conventions agree (everywhere a 2.1 release was decided by data) this is 2.1's value; where they disagree, their pairs
  are disjoint and at most one can pass, so the bin abstains. A switch interval therefore never releases a bin that 2.1
  rejected and never adds a call: it only replaces the named class of the interval track by its candidates' rows, so the
  scaffold keeps the alleles the candidates share (release 2.2 v4; an earlier gate over the candidate sets released more
  bins and lowered the accuracy of the extra calls on stress_n80).

Measured (in-process, fixed final3 panels): the released and abstained bins equal 2.1's exactly (stress_n80 chr2
and chr20, TroMau chr7); interval track bins: TroMau chr7 4,894, stress chr20 2,111; default versus reversed
conventions and default versus iid 1e-9 / value-consistent 1e-12 GL noise release identical unordered painting.

## Release rule

With τ = `PaintingConfig.minimum_release_posterior = 0.9` (a predeclared 10% within-model posterior-error budget),
every sample and bin is released by the first applicable stage (`painting/release.py`; qF and qAF are the faithful
masses of the [decoding](#decoding), release 2.14; qV and qA before):

1. **qF ≥ τ.** Both tracks are released as their Viterbi classes: a singleton anchored class names its row, a
   pooled class (rows identical at every selected site) names its rows without a label, an unanchored class or
   BACKGROUND names none.
2. **Otherwise, qAF ≥ τ** and every track whose Viterbi state belongs to a local key pooling at least two classes in
   the bin is directly supported there. Those tracks are released as LOCAL_ALLELE_EQUIVALENCE: label −1, class =
   the local key, candidate rows = all rows of the classes with that key in the bin. The founder label is not
   identified there, but the alleles are known and the sample's reads observe them. The other track keeps its
   stage-1 release.
3. **Otherwise** both tracks abstain. Without reads in the bin a local-equivalence pair is withheld, as is
   everything below τ.

A track whose candidate set holds more than one class (a switch interval) is released, in a bin that passes stage 1
or 2, as `SWITCH_INTERVAL`: label −1, class −1, candidate rows = the rows of its candidate classes (in a stage-2
bin, of every class sharing a candidate's local key), or none when a candidate is BACKGROUND or an unanchored class
(no row set represents "not in the panel"). The other track is released as usual. The direct-support requirement of
stage 2 applies to every track whose Viterbi state belongs to a multi-class local key, interval tracks included.

| status (uint8) | name | label | candidate rows |
| --- | --- | --- | --- |
| 0 | `INELIGIBLE_NO_EVIDENCE` | −1 | none |
| 1 | `LOW_POSTERIOR_ABSTENTION` | −1 | none |
| 2 | `SINGLETON_NAMED` | the row | the row |
| 3 | `POOLED_EQUIVALENCE` | −1 | the class's rows |
| 4 | `BACKGROUND` | −1 | none |
| 5 | `UNANCHORED_TRAJECTORY` | −1 | none |
| 6 | `LOCAL_ALLELE_EQUIVALENCE` | −1 | rows of the classes sharing the released key in the bin |
| 7 | `SWITCH_INTERVAL` | −1 | rows of the candidate classes of a switch the sample's reads cannot place (none if a candidate is BACKGROUND or unanchored) |

The rule runs once, when the product is made; records decode the released grids without re-deriving them.

## Product

`ChromosomePainting` (`painting/release.py`, record `painting`) holds the contig, the sample ids (every evidence
sample, in evidence order), one `ComponentPainting` per assembled component in chromosome order (components without
selected sites included), one `LabelContinuity` per component and the `PaintingConfig` used. Component HMMs are
independent: every assembly break is an unconditional reset. Per component (N samples, B bins, S states):

| field | type | meaning |
| --- | --- | --- |
| `panel` | `ComponentPanel` | the painted panel (assembly component after label continuity); rows of every grid are its component-local row indices |
| `states` | `PaintingStates` | selected sites (`site_indices` into the panel, bp `positions`), `named_alleles int8[C, Ls]`, `classes` (row tuples), `background_frequency` |
| `bins` | `PaintingBins` | `starts`, `stops` into the selected sites, `centers`, bp `edges int64[B + 1]` |
| `transitions` | `PaintingTransitions` | the label-level transition table per bin edge, with the component's estimated ρ (`switch_rate`), a (`error_entry`), b (`error_exit`) and r (`site_correlation`) |
| `local_keys` | `int32[B, S]` | per-bin allele-equivalence keys |
| `eligible` | `bool[N]` | the sample has an informative site on the component |
| `labels` | `int32[N, 2, B]` | released founder row of a `SINGLETON_NAMED` track, else −1 |
| `classes` | `int32[N, 2, B]` | released class (stage 1) or local key (`LOCAL_ALLELE_EQUIVALENCE`), −1 when abstaining or `SWITCH_INTERVAL` |
| `status` | `uint8[N, 2, B]` | the track status above |
| `direct` | `bool[N, 2, B]` | the released class is called and read-observed at an informative selected site of the bin |
| `member_ids`, `member_sets` | `int32[N, 2, B]`, interned row tuples | candidate founder rows per track and bin; `member_sets[0]` is the empty set |
| `switch_counts` | `int64[N]` | minimum unordered Viterbi switches (the fewer of the two conventions' paths) |
| `packed_emissions` | `float64[N, S(S+1)/2, B]` or None | the upper triangle of the tempered binned log emissions, kept for pedigree preparation |

Views: `chunks(sample)` coalesces consecutive bins whose label, class, status and candidate rows agree on both
tracks into `PaintedChunk(start, end, labels, members)` with a bp interval [start, end) (none for an ineligible
sample; a switch interval is its own chunk); `switch_intervals(sample)` lists the released switch intervals as
`SwitchInterval(track, start, end, rows, before, after)` (bp [start, end) of the interval's bins, its candidate rows,
and the track's member rows just before and after); `site_labels(positions)` gives the released labels at bp
positions (−1 outside the bins; −1 inside a switch interval); `carrier_observed(positions, columns, observed, K)`
marks the founder alleles that a painted carrier (a sample whose released track names the row in its member set)
observes with read depth (export withholds the others, release 2.2). `painting/summary` and the log line count the
released switch intervals and their track bins.

Consumers: pedigree inference rebuilds its source model from `states`, `bins`, `transitions`, the released grids and
the packed emissions (recomputed with the same kernel and divided by the stored T when absent), using the stored ε and
floor, with the label-level table (no panel-error state); the founder refit
reads `site_labels` and the painted panels; family phase reads the bin edges (its phase bins), eligibility, candidate
rows, local keys, trajectory classes and chunks; export and evaluation read the painted panels as founder stage
`painting` (per component in chromosome order: positions, released and inference calls; founder labels are row
indices 0..K−1).

Inside a switch interval no consumer counts the sample as evidence for either candidate: the pedigree's junction
counter sees an unknown label (class −1: compatible with every trajectory, not a callable bin), the founder refit
sees no label, and the family-phase scaffold takes an allele only where every candidate row has the same call (−1
where they differ, sites the sample does not observe by construction).

## Published reports

After each contig is painted (and again when a run resumes a painted contig), `OUTPUT/painting/<contig>/` receives
eight tab-separated tables (`reports/painting.py`). Positions are variant positions, 1-based inclusive
(`FirstVariant1Based`, `LastVariant1Based`), not BED coordinates; founder labels are component-local row indices of
the painted panel.

- `founder_support.tsv`, one row per component and founder row: marker count, called and missing alleles of the
  released calls, break flags, and carrier support. `ObservedCarrierSamples` counts samples whose released named
  track (`SINGLETON_NAMED`) carries the row at a site where the row is called and the sample has positive read
  depth; `CalledSitesWithoutObservedNamedCarrier`, `MinimumObservedCarriers` and `MeanObservedCarriers` summarise
  those carriers over the row's called sites. Pooled candidates (rows named by a `POOLED_EQUIVALENCE` or
  `LOCAL_ALLELE_EQUIVALENCE` track, which the reads cannot separate) are counted separately in
  `ObservedPooledCandidateSamples`, `MeanObservedPooledCandidates` and
  `CalledSitesWithoutObservedNamedOrPooledCandidate`.
- `founder_support_windows.tsv`: the same callability and carrier summaries in consecutive windows of 200 panel
  sites (the last window may be shorter).
- `assembly_search.tsv`: every record of the final assembly's join searches (one row per search event and a closing
  row per search), with the assembly level and the first and last variant of the joined batch.
- `founder_counts.tsv`, one row per component: `FounderRows` and how firmly that count is determined.
  With founder refinement enabled (including the 2.7.1 default), `FounderRows` is the refined panel's row
  count. `FounderCountAmbiguous` is the final count step's call (the converged refinement count pass of the last assembly level,
  which decides the reported count): true when one of its decisions has a bootstrap band containing zero, when it
  cycled, or when a restored best-F panel fails its own deletion test. `CountStepDecisions` lists the decisions
  (kind, founder count, outcome, margin, band). A count cycle (a row added by a search and deleted again) keeps the
  deletion; it is undecided when that deletion is ambiguous and the cycle began with a decided deletion.
  `JoinCountCycles`, `JoinUndecidedCycles` and `JoinAmbiguousCountCalls` count the final assembly's join searches
  over the component, the `FeedbackJoin*` columns those of the local-selection context assemblies, and
  `UndecidedCountCycles` sums the undecided cycles of all three. Empty cells mean no count step or no join search
  covers the component. Without refinement, `FounderRows` still describes the painted panel, but the
  refinement count-call columns are empty when no call was recorded; join diagnostics remain separate.
- `sample_switches.tsv`, one row per sample: `PaintedSwitches` (the painter's switch counts summed over
  components), `PaintedComponents`, the cohort median over painted samples (not split by generation) and the ratio
  to it.
- `sample_segments.tsv` (release 2.2): the released painting per sample, one row per run of bins with one release on
  a track (status, named row, candidate rows), switch intervals included.
- `label_continuity.tsv` (release 2.14): one row per accepted label-continuity move: atomic span, position, rows,
  score gain in switch units, painted switch change, winners, losers, the winners' directions (`WinnersAtoB`,
  `WinnersBtoA`) and `Identity` (`two_way`, or `ambiguous` when all winners switch one way).
- `painting_model.tsv` (release 2.14): one row per component with bins: its span, bins, states, error-capable classes,
  the estimated ρ (`SwitchRatePerMorgan`), a (`ErrorEntry`), b (`ErrorExit`, with `ErrorRunBinsKeepingLabel` = 1/b), r
  (`SiteCorrelation`) and the eligible samples. The log line `[painting] <contig>: copying model rho ...` summarises it.

How to read them. Carrier counts require both a released named track and positive read depth at a called allele;
a homozygous named carrier counts once per sample and site, and pooled candidates are counted once per sample and
site in their own columns, never attributed to an arbitrary row. Missing founder alleles get no support. Related
carriers are not independent lineages, and a called allele without a direct named carrier is not necessarily
wrong. The assembly-search rows describe the particular join searches of the assembly hierarchy (each level's
search precedes that level's founder refinement), not a posterior over final paths: a small scored gap, an
accepted boundary proposal or a reached backstop is a reason to inspect a region, and a large gap among evaluated
proposals cannot certify alternatives that were never generated. A sample far above the cohort median of switches
is one the panel does not explain as a founder mosaic (for example foreign reads or a missing haplotype), or it
marks a panel error. The log names the three samples with the most switches and the components with an ambiguous
final count or undecided count cycles. All eight tables are diagnostics: nothing is filtered, down-weighted, masked
or fed back.

## Implementation and cost

- **Structure.** Per contig, components are processed in chromosome order: the copying model's estimation, label
  continuity, then decoding in sample batches, then the release rule. Numba kernels use the lane's thread budget;
  samples are independent, so the estimation and the decoder reserve their working memory from the run's memory
  governor (as much as fits, from one sample up to one sample per thread; about 40 S'^2 B + 16 S^2 B + 24 Ls bytes
  per sample: structured emissions, Viterbi back-pointers, one forward table and the label-pair masses) and run
  batches of that size; results do not depend on the batch size.
- **Cost per component.** Emissions O(N Ls S(S + 1) / 2) (parallel over samples and unordered pairs), computed once
  per pass; each chain pass (forward-backward or Viterbi) O(N B S'^2): 8 estimation passes, then two Viterbi passes
  (both conventions) and one forward-backward for the release masses. On the simulations painting's decode takes
  about 1.6 to 1.9 times 2.13's. Local keys O(C Ls); track order O(N B S^2) at worst (O(N B S) for single-state
  candidate sets). Label continuity costs O(links N S^3) per search iteration for the gains (per link and sample the
  unaffected identity terms are sorted once and each row pair evaluates only its 4S − 4 affected states) plus a full
  re-decoding with the chain, O(N B S'^2), per distinct candidate checked.
- **Scaling in K.** S = C + 1 ≤ K + 1 and S' ≤ 2S − 1, so the painting HMM is quadratic in the founder count and the
  label-continuity gains are cubic per search iteration; each distinct candidate check re-decodes the N samples
  (O(N B S'^2)), and
  only candidates whose score gain passes the threshold are checked, so with up to links x C(C - 1)/2 screened
  candidates the checks are at worst quartic in K per iteration (observed: at most 8 distinct per component on the
  validation runs, one per move on synthetic K = 30 components). Nothing is exponential in K. Label continuity's
  arrays (per sample: part emissions 8 S^2 (B + links) bytes, plus messages and link scores per search iteration)
  are its dominant memory term at large K; the search reserves them from the run's memory governor before it
  allocates (`painting/label_continuity.py` `relabel_bytes`) and is granted as much as the measured headroom admits
  (release 2.14; before, at most the larger of 32 GiB and a quarter of the run's memory limit): every sample at once
  when the grant
  covers them, else contiguous sample batches of the granted size, which recompute part emissions (about 3x slower,
  identical results). At K = 30, N = 300 and 10,000 bins with 2,000 link points the whole search needs about 242 GB
  at once (2.13's model: about 79 GB); a smaller grant batches.
  A synthetic component with N = 300 samples, 2,000 bins and 200 link points (112 threads) takes, at K = 10 / 20 / 30,
  14 / 51 / 110 s to estimate the copying model, 11 / 38 / 82 s for label continuity and 3 / 10 / 21 s to decode and
  release, with peaks of 10 / 26 / 50 GiB (2.13's penalty model: 6 / 19 / 40 s and 6 / 11 / 20 GiB in all).
- **Packed emissions.** The upper triangle of the emissions is kept for pedigree preparation while the chromosome's
  running total, over components in chromosome order, stays within
  `painting.stage.EMISSION_CACHE_MAX_BYTES = 1073741824` (1 GiB) and the working-memory budget admits it; otherwise
  pedigree recomputes the emissions.
- **Resume.** `identity/painting` is bound before any work and must match on resume; the `painting` record (written
  after `painting/summary`) marks the contig complete, so an interrupted contig repeats label continuity and
  painting together.
- **Determinism.** Every per-sample reduction runs in a fixed order; decoded paths and posteriors do not depend on
  the thread count or the batch size. The model's estimates sum statistics over sample batches, so their unrounded
  values can differ in the last bits with the batch size (the rounded values agreed in the tests).

## Interpretation and limitations

- Founder labels are component-local row indices; components are independent, so a label in one component has no
  relation to the same index in another. Track order (H1/H2) is a gauge fixed by the rule above, not a parental
  identity, and phase is local to a component.
- qF, qAF, qV and qA are posterior masses within the painting model given the founder panel, pointwise per sample
  and bin. Against simulation truth (release 2.14 replays of the 2.13 final panels; scope: sample-bins whose true
  ancestry is constant over the bin, mixed bins excluded; a decoded pair counts as right when its rows carry the true
  haplotypes' alleles in the bin up to max(1, 2%) mismatches, a local-allele event that differs from qF's faithful
  latent event), most wrong pairs fall where no row of the panel carries the true haplotype, which a named-panel
  label cannot correctly represent (std 8,769 of 9,403 wrong decoded pairs over all buckets). Where the panel held
  the truth (the represented-truth stratum), observed error against mean predicted error (1 − qF) by qF bucket on std
  (8 contigs): qF < 0.5 1.4e-2 against 0.64, [0.5, 0.9) 8.0e-3 against 0.26, [0.9, 0.99) 4.0e-3 against 4.0e-2,
  [0.99, 0.999) 2.2e-3 against 3.4e-3, [0.999, 1 − 10^-6) 9.5e-4 against 3.4e-5, and qF ≥ 1 − 10^-6 7.6e-5 (38 of
  497,327) against 3.8e-7; 2.13's qV: about 4e-3 to 5.5e-3 in every bucket from 0.5 to 1 − 10^-6 and 6.3e-4 (584 of
  923,327) against 2e-10 at the top. On these diagnostics qF is closer to its observed errors than 2.13's qV, not
  calibrated: lower in its low buckets (partly the event mismatch: locally identical rows share the pair's mass),
  about equal in [0.99, 0.999) and over-confident above 0.999 (about 30 and 200 times at std; backcross top 3.8e-6,
  f3 1.6e-5, K30 0). The same 2.13 panels served to tune the model and to measure this; real panels may err
  differently. Abstention means the model cannot meet τ, not that the sample lacks ancestry there; it rose with 2.14
  (std 234 to 3,112 of about 945,000 sample-bins).
- In the tested panels and seeds (1e-6 relative jitter of the genotype likelihoods, two seeds, and the convention
  reversal, on the std, f3, backcross, K30 and TroMau panels of the 2.13 final runs), painting on a fixed panel
  released identical products; there, stage differences under jitter came from the panels.
- The copying model (release 2.14). Painted switch events were matched one to one with true ancestry changes within
  3 bins (maximum-cardinality matching; label continuity on). Against 2.14's other fixes on 2.13's prior: std matched
  events 2,166 -> 2,162 of 2,311 true (recall 0.937 -> 0.936) with painted events 2,389 -> 2,334 (precision 0.907 ->
  0.926); f3 886 of 1,007 matched in both, painted 961 -> 959 (precision 0.922 -> 0.924); backcross matched
  1,786 -> 1,784 of 1,809 with unmatched painted events 9 -> 12 (precision 0.995 -> 0.993, recall 0.987 -> 0.986); K30
  equal (20 of 23 matched, 25 painted). Wrong pairs on represented truth std 609 -> 546, backcross 79 -> 67, f3 and
  K30 equal; released pairs whose rows' whole-component founder identity mismatches the truth std 43,958 -> 43,435,
  backcross 102 -> 78. The model is a trade-off, not an improvement on every count: std loses 4 matched events,
  backcross 2. Without any emission adjustment the estimated rate and the panel-error chain give false excursions
  (std precision 0.772, f3 0.692, backcross 0.800): where a row is locally wrong the path escapes it by changing
  label. A fixed temperature (every bin divided by T = 12; the predecessor model) came close to 2.13's precision on
  the simulations (below it on backcross) but lost a sparsely read true short tract in a frozen control (three bins
  of 8 informative sites: no switch, wrong labels), which the design effect recovers exactly; on the simulations' full
  bins the two behave alike. Design-effect sweep without label continuity, r = 0.1 / 0.125 / 0.15: std precision
  0.925 / 0.925 / 0.922 at recall 0.936 / 0.934 / 0.932; f3 precision 0.924 / 0.928 / 0.928 (recall 0.880); backcross
  precision 0.993 / 0.993 / 0.992 at recall 0.986 / 0.984 / 0.981; K30 unchanged; r = 0.1 kept the most recall and
  the fewest represented-truth errors. Without the panel-error state precision falls where the panel has error runs
  (std chr1 0.899 -> 0.855 with T = 12). The number of EM updates mattered little for the constant-temperature
  predecessor (4, 8 and 16 gave the same released decisions on std and backcross); that is sensitivity evidence, not
  convergence. r is a tuned constant: the same 2.13 panels served to choose and to measure it.
- Downstream (copies of the 2.13 final runs repainted and continued to founder refit): pedigree against truth std
  exact parent configurations 37 -> 38 and true edges found 55 -> 57 of 60 (none extra), f3, backcross and K30
  unchanged; TroMau parent assignments and states unchanged for all 116 samples, Tier B release changed for 3. The
  founder refit reads fewer released labels and loses a little: wrong-or-missing alleles std 461,079 -> 461,529,
  backcross 924 -> 978, f3 223,984 -> 224,001, K30 55,717 -> 55,725. Pedigree's source chain is a reduced model (no
  panel-error state; called alleles transmitted unchanged); a parent painted through a panel-error run is not
  down-weighted there.
- BACKGROUND is a haplotype the panel does not represent. A sole discovered haplotype does not force every sample
  to be homozygous for it; missing reads never count as allele matches.
- Painting reads the frozen inference snapshot: completion fills and the founder refit never enter the emissions,
  so carrier-state evidence has no feedback loop.
- Resolution is the painting bin (about 100 selected sites); a change of ancestry inside a bin is placed at one of
  its edges, which family phase and recombination inherit.
- Samples that the sample-quality QC keeps out of founder construction are painted and scored by label continuity
  like every other sample.
- A real crossover inherited by one lineage is relabelled away by label continuity when the decoded carriers of the
  two haplotypes from other lineages that span the link (losers) are at most a tenth as many as the decoded
  recombinant carriers (winners); where none are painted the rule cannot tell it from a label exchange. Label
  continuity is therefore a heuristic for founder identity that can act on weak evidence where few samples carry the
  two rows: on the simulations every accepted move (10 over the saved 2.12 and 2.13 runs) had 3 to 6
  winners and no loser, at links where the two rows differ (not inside identical stretches); against truth, moves
  of one kind improved founder identity (std chr10 -24.8k, chr18 -0.9k, chr20 -16k, K30 m5 -0.5k to -0.8k
  wrong-or-missing alleles) and moves of another worsened it (std chr20 +16.9k in the 2.12 and 2.13 runs, chr1
  +1.8k), with no truth-free feature found to separate them. On TroMau (2.13: 11 moves on 7 contigs, 4 to 19
  winners, no loser) the painted panels agree better with the G0 pseudo-reference over whole components than the
  refined panels (wrong alleles 557,796 -> 540,832; dependent reference, not truth). Label continuity is kept, and
  its moves, winners and losers are reported per component.
