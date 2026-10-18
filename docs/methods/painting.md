# Painting

Painting decodes every sample, on each assembled founder component of a chromosome, as a diploid mosaic of the
component's founder rows plus an explicit BACKGROUND state for haplotypes the panel does not represent, and
releases per sample, track and bin the most specific ancestry statement whose within-model posterior reaches a
fixed threshold. The stage (`painting`, entry point `painting.run_contig`) runs per contig in the contig's lane,
right after the final assembly. Its first step, label continuity, relabels row suffixes of the assembled panel
where two rows exchange founder identity at a link point; the relabelled panel is the one painted, published and
read downstream. Pedigree inference, the [founder refit](founder_refit.md) and [family phase](family_phase.md)
read the painting; [recombination](recombination.md) does not (its bins come from the family-phase product).
Nothing downstream feeds back into the painting or the assembly.

## Inputs and outputs

| | record or setting | content |
| --- | --- | --- |
| reads | `assembly/final` | the final assembly's components as `ComponentPanel`s: positions, released calls (with completion fills), inference calls (the frozen pre-fill snapshot), kept sites, atomic spans, break reasons, count call |
| reads | `evidence` | normalised genotype likelihoods `float64[N, L, 3]` and the positive-depth mask `observed bool[N, L]` of **all** samples, including those the sample-quality QC keeps out of founder construction |
| reads | the contig's genetic map | `[genetic_map] file` and `[genetic_map] rate_cm_per_mb` (fallback rate); see [genetic maps](../genetic_maps.md) |
| writes | `identity/painting` | `PaintingConfig`, the identities of `assembly/final` and `evidence`, the map's digest, code identity |
| writes | `painting/summary` | per-component shapes (founders K, classes C, states S, selected sites, bins, packed emissions kept), eligible sample-component pairs, label-continuity totals |
| writes | `painting` | the `ChromosomePainting` product; its presence marks the contig complete |
| publishes | `OUTPUT/painting/<contig>/` | five diagnostic tables ([published reports](#published-reports)) |

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

Label continuity (`painting/label_continuity.py`) repairs such exchanges with the painting model itself. A suffix
transposition t exchanges the labels of rows a and b at every site from link point j onwards. The set of row
alleles at every site is unchanged and the painting state space is renamed, not changed, so every emission is
unchanged except in the bin that straddles j. For every sample s,

    V_s(t) = max_x [ G_s(x) + L_s(x) + Q_s(t(x)) ],   Q_s = P_s + B_s,

where G_s is the max-product forward message into the bin holding j, L_s and P_s are that bin's emissions on its
sites left and right of j, and B_s is the backward message from that bin. The Hamming transition is invariant when
both of its arguments are renamed, so V_s(t) is the exact best-path score of the relabelled panel: paths
re-optimise on both sides, and a switch anywhere inside a stretch of identical rows vanishes when the relabelling
makes the haplotype continuous.

Search and acceptance use no pedigree, cohort label or truth:

- Link points are atomic-span boundaries strictly inside the component's selected sites. Only singleton anchored
  trajectory classes move; pooled duplicate rows and BACKGROUND keep their labels. A component with fewer than two
  movable rows, no link point, fewer than two bins or no eligible sample is painted unchanged.
- Candidates (link, row pair) are ranked by their summed score gain over the eligible samples and considered only
  when the gain reaches `PaintingConfig.label_continuity_minimum_switch_gain = 2.0` switch units u, where u is the
  median over the component's bin edges of the magnitude of the unnormalised log weight of a one-track change,
  |log θ' + log(1 − θ) − log(S − 1)| ([transitions](#transitions)); one painted switch costs about u.
- A candidate is decided on painted switches exactly as painting counts them: the samples whose switch count it
  can change are re-decoded with the painting Viterbi under the candidate labels. It is accepted when the total
  falls by at least 2 switches and the samples whose count rises (losers) number at most
  `PaintingConfig.label_continuity_maximum_loser_ratio = 0.1` times those whose count falls (winners). Under a
  label exchange the only losers are carriers with a real change of haplotype inside the link interval. Under a
  real founder-to-child crossover inherited by one lineage, every other carrier of the two haplotypes loses a whole
  switch, so the ratio rule keeps such a crossover in the rows even when the recombinant lineage is the majority
  (one unsequenced F1 can parent half of an F2 cohort). Score changes do not classify samples.
- The first acceptable candidate in ranked order is applied, every message is recomputed, and the search repeats
  until no candidate is acceptable. Each accepted move lowers the painted switch total, so the search terminates.
- The relabelled component is kept only if repainting it with the painting functions lowers the total and leaves
  its trajectory classes and selected sites unchanged.

A sample whose best state at the link strictly prefers a state involving neither transposed row, under both the
current and the candidate labels, cannot change its painted switches (barring exact score ties between optimal paths
with different switch counts); only the other samples are re-decoded, and each re-decoded sample is bitwise what a
full decode gives.

Per component the outcome is a `LabelContinuity` record: `reason` (`relabelled`, `no_link_or_movable_row`,
`no_evidence`, `no_move`, `trajectory_classes_changed` or `painted_switches_not_lowered`), link points, movable rows,
u, switch totals before and after, candidate checks, the accepted moves (atomic span, bp position, rows, score gain
in switch units, painted switch change, winners, losers) and, when accepted, the row maps (in atomic span a, painted
row k carries assembled row `row_maps[a, k]`). Released and inference calls are permuted together; the painting
record stores the painted panel as row paths over the assembly's prepared panels, while the assembly records keep the
unrelabelled rows. Label continuity scores all samples, as the painting does. The log reports each contig as
`[label continuity] <contig>: m row-suffix relabellings; painted switches a -> b`.

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

## Transitions

Each track keeps its state between consecutive bins or switches to one of the other S − 1 states
(`painting/transitions.py`). With θ the switch probability of the bin edge and θ' = θ exp(−λ n), where
λ = `PaintingConfig.switch_penalty_per_snp = 1.0` nat per selected site and n = `snps_per_bin`, an ordered pair
moves to

- the same pair with weight (1 − θ)^2,
- each of the 2(S − 1) pairs that change one track with weight θ'(1 − θ) / (S − 1),
- each of the (S − 1)^2 pairs that change both tracks with weight θ'^d / (S − 1)^2, where
  d = `PaintingConfig.double_recomb_factor = 1.5`,

normalised over the S^2 successor states (`PaintingTransitions`: `log_same`, `log_one`, `log_two` per bin edge,
with same + 2(S − 1) one + (S − 1)^2 two = 1). Normalising an edge by one constant changes no Viterbi path or
posterior, and the same normalised table is pedigree's transition model. θ is Haldane's recombination fraction
of the genetic distance d_M (Morgans) between consecutive bin centres on the contig's map,
θ = (1 − e^(−2 d_M)) / 2 (`model.genetic_map.haldane_fraction`, as in family phase, phase polishing and
recombination), and at least 1e-15 (`painting.transitions.MINIMUM_SWITCH_PROBABILITY = 1e-15`), so two bin centres in
a cold map interval keep a negligible switch probability and every transition weight stays finite.
At the defaults one track change costs about 109 nats, so a painted switch needs the evidence of many sites.

## Decoding

Every sample is decoded independently (`painting/hmm.py`):

- **Viterbi.** Exact max-product recursion over the S^2 ordered states, O(S^2) per bin: the best predecessor for
  each of the three Hamming costs comes from the best and second-best entries of every row and column of the
  previous scores and their complement. Exact ties go to the smallest previous flat state; scores are centred per
  bin.
- **Track order.** Emissions are symmetric in track order and the Hamming transition is invariant to swapping both
  tracks, so swapping the whole path suffix after a boundary changes the score only through that one transition.
  Where the ordered and swapped continuations change equally many tracks, both are exact Viterbi paths; the suffix
  is then swapped when the swapped order has strictly fewer allele-visible track changes (a change between classes
  that share a local key below C in either adjacent bin is invisible). This fixes the H1/H2 gauge so that a label
  change between identical rows inside a pooled run is not placed at any bin; unordered states, posteriors and
  releases are unchanged.
- **Release posteriors.** An exact scaled forward-backward pass gives, per bin, qV = the posterior mass of the
  unordered pair of public classes of the Viterbi state (unanchored classes and BACKGROUND share the public UNKNOWN
  identity) and qA = the posterior mass of the unordered pair of local allele keys of the Viterbi state. Keys are
  coarser than public classes, so qA ≥ qV.
- **Eligibility.** A sample is eligible on a component if it has an informative site there: a selected site that is
  observed and whose normalised GL row has a spread (maximum minus minimum) above 16 machine epsilons. Ineligible
  samples abstain in every bin.
- **Direct support.** A track is directly supported in a bin when its Viterbi class is called at an informative
  selected site of the bin.
- **Switch counts.** `switch_counts[s]` is the minimum number of unordered track changes along the Viterbi path (a
  pure exchange of the two tracks is gauge, not a switch), 0 for ineligible samples.

## Release rule

With τ = `PaintingConfig.minimum_release_posterior = 0.9` (a predeclared 10% within-model posterior-error budget),
every sample and bin is released by the first applicable stage (`painting/release.py`):

1. **qV ≥ τ.** Both tracks are released as their Viterbi classes: a singleton anchored class names its row, a
   pooled class (rows identical at every selected site) names its rows without a label, an unanchored class or
   BACKGROUND names none.
2. **Otherwise, qA ≥ τ** and every track whose Viterbi state belongs to a local key pooling at least two classes in
   the bin is directly supported there. Those tracks are released as LOCAL_ALLELE_EQUIVALENCE: label −1, class =
   the local key, candidate rows = all rows of the classes with that key in the bin. The founder label is not
   identified there, but the alleles are known and the sample's reads observe them. The other track keeps its
   stage-1 release.
3. **Otherwise** both tracks abstain. Without reads in the bin a local-equivalence pair is withheld, as is
   everything below τ.

| status (uint8) | name | label | candidate rows |
| --- | --- | --- | --- |
| 0 | `INELIGIBLE_NO_EVIDENCE` | −1 | none |
| 1 | `LOW_POSTERIOR_ABSTENTION` | −1 | none |
| 2 | `SINGLETON_NAMED` | the row | the row |
| 3 | `POOLED_EQUIVALENCE` | −1 | the class's rows |
| 4 | `BACKGROUND` | −1 | none |
| 5 | `UNANCHORED_TRAJECTORY` | −1 | none |
| 6 | `LOCAL_ALLELE_EQUIVALENCE` | −1 | rows of the classes sharing the released key in the bin |

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
| `transitions` | `PaintingTransitions` | the normalised Hamming table per bin edge |
| `local_keys` | `int32[B, S]` | per-bin allele-equivalence keys |
| `eligible` | `bool[N]` | the sample has an informative site on the component |
| `labels` | `int32[N, 2, B]` | released founder row of a `SINGLETON_NAMED` track, else −1 |
| `classes` | `int32[N, 2, B]` | released class (stage 1) or local key (`LOCAL_ALLELE_EQUIVALENCE`), −1 when abstaining |
| `status` | `uint8[N, 2, B]` | the track status above |
| `direct` | `bool[N, 2, B]` | the released class is called and read-observed at an informative selected site of the bin |
| `member_ids`, `member_sets` | `int32[N, 2, B]`, interned row tuples | candidate founder rows per track and bin; `member_sets[0]` is the empty set |
| `switch_counts` | `int64[N]` | minimum unordered Viterbi switches |
| `packed_emissions` | `float64[N, S(S+1)/2, B]` or None | the upper triangle of the binned log emissions, kept for pedigree preparation |

Views: `chunks(sample)` coalesces consecutive bins whose label, class, status and candidate rows agree on both
tracks into `PaintedChunk(start, end, labels, members)` with a bp interval [start, end) (none for an ineligible
sample); `site_labels(positions)` gives the released labels at bp positions (−1 outside the bins).

Consumers: pedigree inference rebuilds its source model from `states`, `bins`, `transitions`, the released grids and
the packed emissions (recomputed with the same kernel when absent), using the stored ε and floor; the founder refit
reads `site_labels` and the painted panels; family phase reads the bin edges (its phase bins), eligibility, candidate
rows, local keys, trajectory classes and chunks; export and evaluation read the painted panels as founder stage
`painting` (per component in chromosome order: positions, released and inference calls; founder labels are row
indices 0..K−1).

## Published reports

After each contig is painted (and again when a run resumes a painted contig), `OUTPUT/painting/<contig>/` receives
five tab-separated tables (`reports/painting.py`). Positions are variant positions, 1-based inclusive
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
  `FounderCountAmbiguous` is the final count step's call (the converged count pass of the last assembly level,
  which decides the reported count): true when one of its decisions has a bootstrap band containing zero, when it
  cycled, or when a restored best-F panel fails its own deletion test. `CountStepDecisions` lists the decisions
  (kind, founder count, outcome, margin, band). A count cycle (a row added by a search and deleted again) keeps the
  deletion; it is undecided when that deletion is ambiguous and the cycle began with a decided deletion.
  `JoinCountCycles`, `JoinUndecidedCycles` and `JoinAmbiguousCountCalls` count the final assembly's join searches
  over the component, the `FeedbackJoin*` columns those of the local-selection context assemblies, and
  `UndecidedCountCycles` sums the undecided cycles of all three. Empty cells mean no count step or no join search
  covers the component.
- `sample_switches.tsv`, one row per sample: `PaintedSwitches` (the painter's switch counts summed over
  components), `PaintedComponents`, the cohort median over painted samples (not split by generation) and the ratio
  to it.

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
final count or undecided count cycles. All five tables are diagnostics: nothing is filtered, down-weighted, masked
or fed back.

## Implementation and cost

- **Structure.** Per contig, components are processed in chromosome order: label continuity, then decoding in
  sample batches, then the release rule. Numba kernels use the lane's thread budget; samples are independent, so
  the decoder reserves its working memory from the run's memory governor (as much as fits, from one sample up to
  one sample per thread; about 20 S^2 B + 24 Ls bytes per sample: emissions, Viterbi back-pointers and one forward
  table) and decodes batches of that size; results do not depend on the batch size.
- **Cost per component.** Emissions O(N Ls S(S + 1) / 2) (parallel over samples and unordered pairs); Viterbi and
  forward-backward O(N B S^2) each, the Viterbi holding `int32` back-pointers [batch, B, S^2]; local keys
  O(C Ls); track order O(N B). Label continuity costs O(links N S^3) per search iteration for the gains (per link
  and sample the unaffected identity terms are sorted once and each row pair evaluates only its 4S − 4 affected
  states) plus the re-decoding of affected samples per candidate check.
- **Scaling in K.** S = C + 1 ≤ K + 1, so the painting HMM is quadratic in the founder count and the label-continuity
  gains are cubic per search iteration; each candidate check re-decodes at most N samples (O(N B S^2)), and only
  candidates whose score gain passes the threshold are checked. Nothing is exponential in K. Label continuity's
  arrays (per sample: part emissions 8 S^2 (B + links) bytes, plus messages and link scores per search iteration)
  are its dominant memory term at large K; the search reserves them from the run's memory governor before it
  allocates (`painting/label_continuity.py` `relabel_bytes`): every sample at once when that fits within
  `painting.label_continuity.LABEL_CONTINUITY_MAX_BYTES = 34359738368` (32 GiB) or a quarter of the run's memory
  limit, else contiguous sample batches of the granted size, which recompute part emissions (slower, identical
  results).
- **Packed emissions.** The upper triangle of the emissions is kept for pedigree preparation while the chromosome's
  running total, over components in chromosome order, stays within
  `painting.stage.EMISSION_CACHE_MAX_BYTES = 1073741824` (1 GiB) and the working-memory budget admits it; otherwise
  pedigree recomputes the emissions.
- **Resume.** `identity/painting` is bound before any work and must match on resume; the `painting` record (written
  after `painting/summary`) marks the contig complete, so an interrupted contig repeats label continuity and
  painting together.
- **Determinism.** Every per-sample reduction runs in a fixed order; results do not depend on the thread count or
  the batch size.

## Interpretation and limitations

- Founder labels are component-local row indices; components are independent, so a label in one component has no
  relation to the same index in another. Track order (H1/H2) is a gauge fixed by the tie rule above, not a
  parental identity, and phase is local to a component.
- qV and qA are posterior masses within the painting model given the founder panel; they are not calibrated
  against truth. Abstention means the model cannot meet τ, not that the sample lacks ancestry there.
- BACKGROUND is a haplotype the panel does not represent. A sole discovered haplotype does not force every sample
  to be homozygous for it; missing reads never count as allele matches.
- Painting reads the frozen inference snapshot: completion fills and the founder refit never enter the emissions,
  so carrier-state evidence has no feedback loop.
- Resolution is the painting bin (about 100 selected sites); a change of ancestry inside a bin is placed at one of
  its edges, which family phase and recombination inherit.
- Samples that the sample-quality QC keeps out of founder construction are painted and scored by label continuity
  like every other sample.
- A real crossover inherited by one lineage is relabelled away by label continuity only when the other carriers of
  the two haplotypes are at most a tenth as many as the recombinant carriers; the painting cannot then tell it
  from a label exchange.
