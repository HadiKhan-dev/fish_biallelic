# Recombination

The recombination stage (`recombination`, entry point `recombination.stage.run`) estimates where each observed
meiosis changed transmitted parental homologue and aggregates the meioses into missing-aware recombination maps. It
runs genome-wide after [family phase](family_phase.md), one chromosome per worker, and reads only the family-phase
product, the released parent edges and the genetic map. Family phase infers every child's inheritance as meiosis
chains, one per linked homologue, inside its single generative model, and releases each chain's posterior per
painting bin. The stage follows those chains: it reports working expected crossovers (Haldane interpolations
conditional on the called endpoint states), called crossover intervals and followed exposure separately, together with a truth-free detector of reference-order junctions. Everything is
conditional on the final family-phase state and the fixed inferred pedigree; nothing feeds back upstream.

## Inputs and outputs

| | record or setting | content |
| --- | --- | --- |
| reads | `family_phase` | the meiosis chains (`chain_source`, `chain_support`, `chain_reset`, one row per parent edge), positions, component ids, painting-bin edges per component, the identity of the edges it was conditioned on |
| reads | `ParentEdges` | the supported Tier-B parent → child edges (must be the edges the family phase used; the chain rows follow their order) |
| reads | the contig's genetic map | genetic coordinates of the chain's bins; see [genetic maps](../genetic_maps.md) |
| reads | `[recombination] order_breakpoints` | optional reference-order breakpoint TSV |
| writes | `identity/recombination` | `RecombinationConfig` (breakpoint rows included), the identity of `family_phase`, the edges, the map's digest, code identity (recombination and family phase packages) |
| writes | `recombination` | the `RecombinationProduct` per contig; run record `recombination` (summaries) and `complete/recombination` |
| publishes | `OUTPUT/recombination/` | per-contig tables and plot, genome-wide junction and summary tables ([published tables](#published-tables)) |

The other `RecombinationConfig` fields are internal model constants ([configuration reference](../configuration.md)).

## Meiosis chains

In family phase, a homologue of a child with a released slot-k parent copies one of that parent's two homologues,
switching between them along the chromosome with the Haldane probability of the genetic distance between painting-bin
centres, or is not a copy (a painting or panel error, conversion or mutation, or a false edge). For every parent
edge, the release computes the child's block posterior with every other sample at its converged state, under both tie
conventions ([family phase](family_phase.md#release)), and publishes per painting bin:

- `chain_source`: the parent homologue (0 or 1, in the parent's released homologue order) that both conventions give
  a posterior of at least `FamilyPhaseConfig.phase_posterior` (0.99), else -1 (a released bin); `chain_support` is
  the smaller of the two posteriors;
- `chain_reset`: no inheritance is followed across a component's first bin, a meiosis reset (protected intervals,
  where the switch probability is 1/2) or the parent's or the child's released phase-set start (the parent's
  homologue order is not released across it).

The stage follows each chain from reset to reset; supplied order breakpoints add resets (below). Within a followed
stretch:

- a **crossover is called** between two consecutive released bins whose parental homologues differ. Its interval
  joins the two bins' centres (the model places changes between bin centres; `left_origin_probability` and
  `right_origin_probability` are the two bins' supports, so the joint probability of the two released homologues
  is at least their sum minus one: a bound within each conditional block model, not a calibrated biological
  probability). Its reported interval holds both bins, since the change can lie anywhere between
  the last site of the old homologue and the first site of the new one. An even number of crossovers between two
  released bins is not identifiable;
- **exposure** runs from the first to the last released bin, and the callable and informative spans are those
  stretches.

There is no maximum gap between consecutive released bins: a change of homologue across a long stretch without a
released bin (parents homozygous there, or children without reads) is called, with a reported interval spanning the
stretch, and its expected count is spread over the stretch by the input map. Such calls are honest about the parity
but poorly localised (on the TroMau cross, calls across stretches of 12-14 Mb occur on chr3 and chr12); the release
2.x decoder restarted at gaps above 1 Mb and made no call there.

Marker counts per meiosis are its released bins, its followed bins and, among them, the bins without a released
homologue.

## Conditional map

Map bins are `RecombinationConfig.bin_bp = 1000000` bp wide from the first site of the chromosome and split at
component (segment) edges, so no event or exposure crosses an assembly break (`recombination/map.py`). Per meiosis
and interval between the centres of consecutive released bins of a followed stretch (a bin's centre is the mean
position of its sites, as in family phase's transitions; `bin_centers`), with genetic length λ:

- expected crossovers: E[N | even endpoint parity] = λ tanh λ and E[N | odd parity] = λ / tanh λ (Poisson events,
  Haldane parity), by the released parity of the two bins; the mass is spread over the interval in proportion to
  the input map's genetic mass (uniformly in bp without a map);
- exposure: one meiosis per bp of the interval (observable meiosis-bp).

Where the input map gives an interval no genetic length, an odd parity there still counts one crossover (the λ → 0
limit of λ / tanh λ), spread uniformly in bp: the data override the map's zero prior. Called crossovers are binned the
same way, so no call is lost from the map while its exposure and event row remain.

Expected counts are Haldane bridge interpolations conditional on the called parental-homologue states at released
bins. Endpoint calls are treated as fixed, so these counts are not posterior means under the complete family model:
endpoint uncertainty and further evidence within an interval are not propagated, as in the release 2.x decoder.
- Two released bins with different homologues show an odd number of changes between them, not every change, and the
  calls can be wrong: a change can also come from a wrong edge (a related non-parent can fit different tracts), a
  parental phase or panel identity error, or a short conversion.
- Exposure denotes the followed spans between released markers, not independent read support throughout them. A
  released bin can owe its confidence to the transition prior propagated from informative neighbours, and splitting
  an uninformative stretch at such bins counts less of its even-parity (double-crossover) mass than one interval
  would (λ tanh λ grows faster than linearly).
- Calls and maps remain conditional on the inferred pedigree and parental phase.

The chain's own pairwise switch posteriors are not used. In the family model a homologue that leaves the copy states
(not a copy) re-enters either parental homologue with probability 1/2, so the posterior routes part of a change of
homologue through not-a-copy wherever that costs about as much as a crossover, which is often near one. Copy-to-copy
switch posteriors then miss those changes (on the TroMau cross they accounted for fewer than two thirds of the called
changes), and masking not-a-copy bins removes nearly all crossover-associated intervals while keeping almost all
exposure.

Summed over meioses, each bin reports `expected_crossovers` (the working count above; the field keeps its name),
`exposure_meiosis_bp`, the expected rate
`rate_cM_per_Mb` = 10^8 × expected crossovers / exposure (NaN without exposure, which is not a measured zero
rate), `effective_meioses` = exposure / bin width, `covered_bp` (the union of all meioses' informative spans),
`cM_on_covered_bp` = rate × covered bp / 10^6 and its cumulative sum over bins (`cumulative_observed_cM`, gaps
omitted). Called crossovers are reported separately: `crossover_count` (each call spread over its interval by
genetic mass, for binning only), `called_exposure_meiosis_bp` (callable spans) and `called_rate_cM_per_Mb`.

## Reference-order breakpoints

The map assumes that reference order is meiotic order. Where a reference segment is misplaced or inverted, a
meiosis changes transmitted homologue at the junction whenever it has an odd number of true crossovers between the
two loci the reference joins, with probability (1 − e^(−2d))/2 for loci d Morgans apart: about half of all meioses
for unlinked sides. These are real transmission changes in reference coordinates, not crossovers at that position.

`[recombination] order_breakpoints` names a tab-separated file with the header line `chrom start end reason` (`#`
starts a comment; the reason may be empty). Positions are 1-based inclusive bp, and the order discontinuity lies
somewhere inside [start, end] (for example the contig gap at the junction). A row whose reason starts with
`reversed_segment` declares an inverted reference segment instead: both of its ends act as breakpoints and calls
inside it are flagged `in_reversed_segment` (they stay in reference coordinates). Before anything else
(`split_at_breakpoints` in `recombination/junctions.py`), every interval is widened to the painting bins it overlaps,
those sites leave the map axis, and the component is split there into segments; a chain bin whose sites leave the
axis or fall in two segments is not followed, and every segment change resets the chains, so no call or exposure
crosses a supplied breakpoint, at the cost of those bins' exposure. Calls and spans are reported with their assembly component. The rows' content (not the file path) enters
the stage identity; there is no default file, since a breakpoint list is reference-structure curation, not something
the pipeline infers. The upstream stages do not use the file: their panels, paintings and frames still assume
reference order across it (family phase reads its own `[family_phase] protected_intervals`, which may name the same
file, and resets its chains there). A supplied breakpoint also hides any genuine structural polymorphism there, and decoding a segment on its own
can expose junctions inside it.

## Junction detector

Without a breakpoint file the stage still reports a truth-free junction detector (`recombination/junctions.py`); it
never masks anything. Called crossovers whose reported intervals overlap are chained into loci, and a locus with
calls of at least `RecombinationConfig.junctions.minimum_parents = 3` parents is a candidate. A meiosis is decidable
across a locus when it has a call there or a callable span covering the locus's calls; y is the parity of its calls
in the locus, modelled as y = s XOR f, with s ~ Bernoulli(q) a real change of transmitted homologue and f (prior
1/2, one per parent, shared by its children) a frame flip of that parent.

- **Mixing statistic.** Ordinary recombination, hotspots included, bounds q by a ceiling h: a point hotspot
  (`RecombinationConfig.junctions.hotspot_switch_probability = 0.05`) combined (XOR) with the background Haldane
  parity across the locus on the map. The statistic is the profile log likelihood ratio of free q (a 500-point grid
  on [0.001, 0.5]; q and 1 − q are equivalent under the flip) against q ≤ h, so all-or-none parents (frame flips)
  and hotspot-like single calls give no evidence.
- **Pairing statistic.** The two ends of a displaced or inverted segment switch in the same meioses. For every pair
  of loci on the chromosome, y at one locus predicted from the other up to a discordance probability (50 points on
  [0.005, 0.25]; the floor keeps one discordant meiosis from erasing the evidence) is compared with independent
  switching, the weaker of the two directions is taken, and each locus keeps its best positive pairing
  (`pairing_log_lr`); the flag uses it minus log(number of locus pairs tested on the chromosome, `pairing_tests`).
- **Flag.** `junction_log_lr` = max(mixing, adjusted pairing, 0); a locus is flagged when it reaches
  `RecombinationConfig.junctions.flag_log_likelihood_ratio = 4.605170185988092` (ln 100). The interpretation is
  `junction` for a flagged locus, `parent_frame_flips` when all decidable children of some parent with at least two
  of them switch and the flip-adjusted switch fraction stays within the ceiling, and `recombination_or_unresolved`
  otherwise. With a breakpoint file, each locus also names the nearest supplied breakpoint and its distance.

The likelihood ratios are conditional on the calls (no reads), assume independent meioses given the parents' flips,
and are not calibrated posterior odds; power is limited by the number of informative meioses and by q itself
(junctions joining loci a few cM apart look like hotspots). Use the table to find candidate loci for curation, not
as a breakpoint list by itself.

## Published tables

After the stage (and after a rerun), `OUTPUT/recombination/` receives, per contig:

| file | rows and columns |
| --- | --- |
| `<contig>.map.csv` | one row per map bin [`left_bp`, `right_bp`): `crossover_count`, `exposure_meiosis_bp`, `expected_crossovers`, `called_exposure_meiosis_bp`, `called_rate_cM_per_Mb`, `covered_bp`, `effective_meioses`, `rate_cM_per_Mb`, `cM_on_covered_bp`, `cumulative_observed_cM_at_right` |
| `<contig>.crossovers.csv` | one row per called crossover: `parent`, `child`, `parent_slot`, `left_bp`, `right_bp` (the centres of the flanking released chain bins), `left_origin_probability`, `right_origin_probability` (their supports), `joint_endpoint_probability_lower_bound` (max(0, left + right − 1)), `component_id`, `reported_left_bp`, `reported_right_bp`, `in_reversed_segment` |
| `<contig>.coverage.csv` | informative spans (followed chain stretches) per meiosis: `parent`, `child`, `parent_slot`, `left_bp`, `right_bp`, `component_id` |
| `<contig>.called_coverage.csv` | callable spans per meiosis (the same stretches), same columns |
| `<contig>.meioses.csv` | one row per meiosis: `released_chain_bins`, `followed_chain_bins`, `unresolved_chain_bins`, `called_crossovers`, `expected_crossovers`, `exposure_meiosis_bp` |
| `<contig>.junctions.csv` | one row per candidate locus: bounds (`left_bp`, `right_bp`, `calls_left_bp`, `calls_right_bp`), `calls`, `parents_with_calls`, `odd_meioses`, `decidable_meioses`, `switch_fraction`, `flip_adjusted_switch_fraction`, `hotspot_ceiling`, `mixed_parents`, `all_or_none_parents`, per-parent `k/n` counts (`parents`) and switching `children`, `mixing_log_lr`, `paired_locus`, `paired_odd_meioses`, `pairing_log_lr`, `pairing_tests`, `junction_log_lr`, `flagged`, `interpretation`, `nearest_supplied_breakpoint`, `breakpoint_distance_bp` |
| `<contig>.png` | expected and called rates (cM/Mb) and effective meioses along the chromosome |

and genome-wide `junctions.csv` (every contig's loci) and `summary.csv`, one row per contig: `meioses`,
`observable_meioses` (meioses with a callable span), `crossovers`, `expected_crossovers`,
`unresolved_bins` (summed over meioses), `exposure_meiosis_bp`, `covered_bp`, `cumulative_observed_cM`,
`uncovered_bins`, `order_breakpoints`, `order_breakpoint_masked_sites`, `junction_loci`, `junction_loci_flagged`, `family_phase_converged` and
`elapsed_seconds`. Sample names identify parents and children; `parent_slot` is the edge's slot (0 for the child's
Parent1, 1 for Parent2).

## Rerunning the stage

`haplotypes recombination RUN` runs this stage again from a run's records and republishes the tables: it loads the
run's resolved configuration and accepts only `--set recombination.KEY=VALUE`, `--order-breakpoints FILE` (the same
as `[recombination] order_breakpoints`) and `--cores N`, so every other setting is the run's own; it reads the
stored pedigree edges and family-phase products. Each contig's `identity/recombination` is bound once in its
checkpoint file, from the recombination settings (breakpoint rows included), the upstream record identities, the
map and the code identity. A contig without a recombination record is mapped with the given settings; a contig
whose record was made with the same identity is reused as it is; a contig whose record was made with other
recombination settings (another breakpoint file content, a changed recombination or family-phase code) refuses the change with a checkpoint mismatch naming `recombination`, and another output root is
needed. Records are append-only: a stored record is never replaced.

## Implementation and cost

- **Structure.** One chromosome per worker of a chromosome executor sharing the stage's CPU budget. Each reports a
  byte estimate (twice the decoded family-phase product, plus 2 GiB); the memory governor paces new work by measured
  memory pressure. The chain posteriors are computed in family phase's release, with the same forward-backward as
  the genotypes and phase; the stage only follows them.
- **Cost.** Independent of the founder count K here: O(E M) per chromosome for E parent edges and M painting bins,
  plus the map aggregation. The chain posteriors are marginals of family phase's existing release forward-backward
  and add no pass of their own. The junction pairing costs O(loci² × parents × grid) with tabulated mixture terms.
- **Resume.** `identity/recombination` is bound per contig before work and checked; a contig with a
  `recombination` record is not recomputed.
- **Determinism.** Meioses are independent and every step runs in a fixed order.

## Interpretation and limitations

- Every output is conditional on the final family-phase state and on the fixed inferred pedigree (the released
  Tier-B edges): a chain's posterior holds every other sample at its converged state. It is not a calibrated joint
  uncertainty over pedigree, phase and recombination. Meioses with an unreleased or unsampled parent are not
  observed.
- Expected crossovers, called crossover intervals and observable exposure are separate outputs. A called
  crossover is a change of released parental homologue between consecutive released chain bins; an even number of
  crossovers between them is not identifiable, and the chain resolves changes to painting bins, not sites.
- A change of released parental homologue across a not-a-copy stretch (a local panel or painting mismatch, or a bin
  holding the crossover itself) is called: the child's inheritance changed between the released bins, whatever the
  chain's route between them. A false parent edge, or a related non-parent fitting different tracts, would be read
  as crossovers.
- The input map is a process prior, not ground truth. Informative data can support departures from a positive
  prior, but rates in weakly observed regions remain prior-sensitive; a zero prior is overridden only where released
  endpoints differ across it (the λ → 0 limit, spread uniformly in bp), and structure entirely between released bins
  is allocated by the input map, not learned. Bins without
  exposure have NaN rates, not zero recombination.
- Reference-order junctions appear as crossover pile-ups unless a breakpoint file masks them; the junction table
  points to candidates but never changes calls or maps.
