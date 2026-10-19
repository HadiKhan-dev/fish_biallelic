# Recombination

The recombination stage (`recombination`, entry point `recombination.stage.run`) estimates where each observed
meiosis changed transmitted parental homologue and aggregates the meioses into missing-aware recombination maps. It
runs genome-wide after [family phase](family_phase.md), one chromosome per worker, and reads only the family-phase
product (decoder-frame calls, painting-bin edges, coverage), the released parent edges, the genetic map and,
optionally, the evidence's read-depth mask. Per chromosome it fits optional shared-family orientation corrections,
decodes every parent → child meiosis with a selector × orientation-artifact HMM, and reports posterior expected
crossovers, called crossover intervals and observable exposure separately, together with bin-honest call intervals
and a truth-free detector of reference-order junctions. Everything is conditional on the final phase and the fixed
inferred pedigree; nothing feeds back upstream.

## Inputs and outputs

| | record or setting | content |
| --- | --- | --- |
| reads | `family_phase` | calls with their displayed and decoder frames (meioses are decoded in the decoder frame), positions, component ids, painting-bin edges per component, each sample's painted coverage, the identity of the edges it was conditioned on |
| reads | `ParentEdges` | the supported Tier-B parent → child edges (must be the edges the family phase used) |
| reads | `evidence` (positions and `observed` only) | the positive-depth mask, when `read_free_tract_sites` > 0 |
| reads | the contig's genetic map | Haldane switch probabilities and genetic coordinates; see [genetic maps](../genetic_maps.md) |
| reads | `[recombination] shared_family_evidence` | fit shared-family orientation corrections before decoding (default on) |
| reads | `[recombination] order_breakpoints` | optional reference-order breakpoint TSV |
| writes | `identity/recombination` | `RecombinationConfig` (breakpoint rows included), the identities of `family_phase` (and of `evidence` when read-free tracts are excluded), the edges, the map's digest, code identity (recombination and family phase packages) |
| writes | `recombination` | the `RecombinationProduct` per contig; run record `recombination` (summaries) and `complete/recombination` |
| publishes | `OUTPUT/recombination/` | per-contig tables and plot, genome-wide junction and summary tables ([published tables](#published-tables)) |

The other `RecombinationConfig` fields are internal model constants ([configuration reference](../configuration.md)).

## Observable coverage

An observation of meiosis (parent, child, slot) at a site requires the parent to be heterozygous with both decoder
alleles called, the child's track in that slot to be called, the site to lie in an assembly component, and both
samples to be covered there. A sample's coverage is the span of every component in which it was painted
(`family_phase` coverage). Final calls inside a long run of sites without reads are imputed from the painting, not
observed: a sample's run of at least `RecombinationConfig.read_free_tract_sites = 1000` consecutive zero-depth sites
(from the evidence's positive-depth mask) is excluded from its coverage, so its meioses, as parent or child, have no
observation, exposure or called crossover inside or across that run, in both the shared-orientation fit and the
decoder. Shorter zero-depth runs remain observations; 0 disables the rule.

Every chain below restarts at a boundary of the parent's or the child's coverage, at every component (or segment, see
[order breakpoints](#reference-order-breakpoints)) change, and wherever consecutive informative sites lie more than
`RecombinationConfig.maximum_gap_bp = 1000000` bp apart; the stretches between restarts are called runs.

## Shared-family orientation fit

A sample's decoder-frame orientation can be wrong over a tract (a phase or painting error); every meiosis it takes
part in, as parent or child, then shows the same apparent switches. With `[recombination] shared_family_evidence`
(default true) the stage fits orientation corrections before decoding (`recombination/shared_orientation.py`):

- **Candidates.** For every meiosis, a two-state screening chain on genetic coordinates (Haldane transitions,
  emissions with the copy error, restarts at run boundaries) gives the posterior of the transmitted homologue at
  each informative site. Runs of supported states (posterior at least
  `RecombinationConfig.shared_orientation.candidate_support = 0.9`) propose tracts: a run between two opposite
  supported runs, or the run between a run boundary of the individual (chromosome end, component change,
  coverage gap or oversized marker gap) and its nearest supported switch. A tract is a candidate for the parent and
  for the child of the meiosis if that sample takes part in at least
  `RecombinationConfig.shared_orientation.minimum_incident_edges = 2` meioses, the tract lies inside one of the
  sample's runs, and it is at most `RecombinationConfig.shared_orientation.maximum_candidate_bp = 1000000` bp long.
- **Objective.** The sum over meioses of the exact marginal log likelihood of the selector × artifact HMM below
  (relative to neutral observations), plus, per sample, the log prior of its orientation path under an asymmetric
  orientation-error process that enters at `RecombinationConfig.shared_orientation.error_rate = 1e-08` per bp and
  lasts `RecombinationConfig.shared_orientation.error_mean_bp = 50000.0` bp on average (restarting at uncovered
  sites).
- **Ascent.** Candidates are visited in descending order of their sample's number of meioses (well-supported shared
  parents first), for at most `RecombinationConfig.shared_orientation.maximum_sweeps = 4` sweeps; a flip of the
  sample's orientation on the tract is accepted when it raises the objective by more than
  `RecombinationConfig.shared_orientation.minimum_objective_gain = 2.0` nats, and a sweep without an accepted flip
  ends the fit (`converged`). Reaching the sweep limit leaves the map conditional on non-converged shared paths and
  is reported as a warning.

The accepted flips reorder the calls given to the decoder; the family-phase product is not overwritten. The fit is
a conditional model comparison, not proof: synchronised true crossovers in several children can be confounded with
one parental orientation error.

## Per-meiosis decoding

Each meiosis is decoded independently (`recombination/decoder.py`). The observation at an informative site is which
parental homologue (decoder frame) the child's allele matches. The hidden state pairs the transmitted homologue S
with an orientation artifact B:

- S switches between consecutive informative sites with the Haldane probability (1 − e^(−2d))/2 of their genetic
  distance d on the contig's map (the map's fallback rate without one);
- B is a correlated artifact process that enters at `RecombinationConfig.artifact_rate = 1e-08` per bp and leaves
  after `RecombinationConfig.artifact_mean_bp = 50000.0` bp on average (asymmetric two-state continuous-time chain,
  started at its stationary distribution);
- the observed homologue is S XOR B, read with error `RecombinationConfig.copy_error = 0.01`.

A scaled forward-backward pass gives, per informative site, the posterior of the transmitted homologue, the
posterior of no artifact, and the posterior probability of a homologue switch in the interval before it. A site is
supported when both its homologue support max(P(S = 0), P(S = 1)) and its no-artifact posterior reach
`RecombinationConfig.minimum_origin_probability = 0.98`. Consecutive supported sites of one run (no reset between
them, at most `maximum_gap_bp` apart) bound a callable span; when their most probable homologues differ, a crossover
is called in (left, right], with left and right the positions of the flanking supported sites. Each meiosis records
its called crossovers (positions, the homologue support at both ends, assembly component), callable spans,
informative spans (between consecutive informative sites of a run) and marker counts (informative, supported,
artifact-uncertain).

## Conditional map

Map bins are `RecombinationConfig.bin_bp = 1000000` bp wide from the first site of the chromosome and split at
component (segment) edges, so no event or exposure crosses an assembly break (`recombination/map.py`). Per meiosis
and interval between consecutive informative sites of a run, with genetic length λ:

- expected crossovers: E[N | even endpoint parity] = λ tanh λ and E[N | odd parity] = λ / tanh λ (Poisson events,
  Haldane parity), mixed by the posterior switch probability of the interval; the mass is spread over the interval
  in proportion to the input map's genetic mass (uniformly in bp without a map);
- exposure: one meiosis per bp of the interval (observable meiosis-bp).

Summed over meioses, each bin reports `expected_crossovers`, `exposure_meiosis_bp`, the posterior-mean rate
`rate_cM_per_Mb` = 10^8 × expected crossovers / exposure (NaN without exposure, which is not a measured zero
rate), `effective_meioses` = exposure / bin width, `covered_bp` (the union of all meioses' informative spans),
`cM_on_covered_bp` = rate × covered bp / 10^6 and its cumulative sum over bins (`cumulative_observed_cM`, gaps
omitted). Called crossovers are reported separately: `crossover_count` (each call spread over its interval by
genetic mass, for binning only), `called_exposure_meiosis_bp` (callable spans) and `called_rate_cM_per_Mb`.

## Bin-honest reported intervals

Painting and phase are rigid inside a painting bin, so a change of homologue inside a bin is placed at one of its
edges. A called crossover whose flanking supported sites straddle painting-bin edges is therefore also reported
with the bins on both sides of those edges (`reported_left_bp`, `reported_right_bp`, inclusive), clipped to the
call's decoded segment, when that widened span is at least `RecombinationConfig.junctions.snapped_bin_minimum_bp = 0`
bp (always, at the default). Calls, counts and the map use the calls themselves, not the reported widths. Bin edges
come from the family-phase product.

Released frame ambiguity (release 2.2). Family phase releases, per sample and site, whether its decoder frame is
undetermined (`frame_undetermined` bit 0: the two extreme tie placements of the sample's phase chain disagree there,
e.g. a frame switch the data cannot place inside a stretch without informative sites). Such a stretch holds no
informative observation of any meiosis through that frame (a parent's undetermined stretch has no heterozygous
decoder site; a child's has no site whose transmitted allele depends on its frame), so moving the switch inside it
changes no observation: called crossovers, their flanking supported sites and their reported intervals are invariant
to where the family phase places such a switch, which is what makes them interval-censored by construction. The
convention-reversal test (`HAPLOTYPES_TIE_REVERSE=1`, which places such switches at the rightmost tied position)
checks this.

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
those sites leave the map axis, and the component is split there into segments; every reset rule keys on segment
changes, so no call, exposure or shared-orientation tract crosses a supplied breakpoint, at the cost of those bins'
exposure. Calls and spans are reported with their assembly component. The rows' content (not the file path) enters
the stage identity; there is no default file, since a breakpoint list is reference-structure curation, not something
the pipeline infers. The upstream stages do not use the file: their panels, paintings and frames still assume
reference order across it (family phase reads its own `[family_phase] protected_intervals`, which may name the same
file). A supplied breakpoint also hides any genuine structural polymorphism there, and decoding a segment on its own
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
| `<contig>.crossovers.csv` | one row per called crossover: `parent`, `child`, `parent_slot`, `left_bp`, `right_bp` (flanking supported sites), `left_origin_probability`, `right_origin_probability`, `joint_endpoint_probability_lower_bound` (max(0, left + right − 1)), `component_id`, `reported_left_bp`, `reported_right_bp`, `in_reversed_segment` |
| `<contig>.coverage.csv` | informative spans per meiosis: `parent`, `child`, `parent_slot`, `left_bp`, `right_bp`, `component_id` |
| `<contig>.called_coverage.csv` | callable spans (between supported sites) per meiosis, same columns |
| `<contig>.meioses.csv` | one row per meiosis: `informative_markers`, `supported_origin_markers`, `orientation_artifact_uncertain_markers`, `called_crossovers`, `expected_crossovers`, `exposure_meiosis_bp` |
| `<contig>.junctions.csv` | one row per candidate locus: bounds (`left_bp`, `right_bp`, `calls_left_bp`, `calls_right_bp`), `calls`, `parents_with_calls`, `odd_meioses`, `decidable_meioses`, `switch_fraction`, `flip_adjusted_switch_fraction`, `hotspot_ceiling`, `mixed_parents`, `all_or_none_parents`, per-parent `k/n` counts (`parents`) and switching `children`, `mixing_log_lr`, `paired_locus`, `paired_odd_meioses`, `pairing_log_lr`, `pairing_tests`, `junction_log_lr`, `flagged`, `interpretation`, `nearest_supplied_breakpoint`, `breakpoint_distance_bp` |
| `<contig>.png` | posterior-mean and called rates (cM/Mb) and effective meioses along the chromosome |

and genome-wide `junctions.csv` (every contig's loci) and `summary.csv`, one row per contig: `meioses`,
`observable_meioses` (meioses with a callable span), `crossovers`, `expected_crossovers`,
`artifact_uncertain_markers`, `exposure_meiosis_bp`, `covered_bp`, `cumulative_observed_cM`, `uncovered_bins`,
`read_free_tracts`, `read_free_sample_sites`, `shared_family_evidence`, `shared_orientation_converged`,
`shared_orientation_corrected_samples`, `shared_orientation_accepted_moves`, `order_breakpoints`,
`order_breakpoint_masked_sites`, `junction_loci`, `junction_loci_flagged`, `family_phase_converged` and
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
recombination settings (another breakpoint file content, the shared-family switch, a changed recombination or
family-phase code) refuses the change with a checkpoint mismatch naming `recombination`, and another output root is
needed. Records are append-only: a stored record is never replaced.

## Implementation and cost

- **Structure.** One chromosome per worker of a chromosome executor; the number of concurrent chromosomes is
  limited by a per-chromosome byte estimate (six decoded family-phase products, the read-depth mask when read-free
  tracts are excluded, and 2 GiB). Inside a chromosome, the decoder and the candidate screening run in parallel over
  meioses.
- **Cost.** Independent of the founder count K. Decoding is O(L) per meiosis (a four-state chain), so O(E L) per
  chromosome; outputs grow per meiosis, never as sample × site × state arrays. The shared-orientation fit builds
  lazy-flip segment trees of the meioses' scaled 4 × 4 interval products and of the samples' orientation priors in
  O(E L), after which each trial and each accepted flip costs O(log J) per incident meiosis (J tree leaves). The
  trees need a positive artifact process and strictly increasing genetic coordinates; with a flat stretch of a
  supplied map (or without an artifact process) the scaled streaming recurrences score each trial in O(L) per
  incident meiosis instead, and markers joined by zero-probability transitions are contracted into one chain step
  in the screening posterior. The junction pairing costs O(loci^2 × parents × grid) with tabulated mixture terms.
- **Resume.** `identity/recombination` is bound per contig before work and checked; a contig with a
  `recombination` record is not recomputed.
- **Determinism.** Meioses are independent and every recurrence runs in a fixed order. Trial flips are scored in
  batches as wide as the thread count but decided one at a time in the fixed greedy order, with scores made stale by
  an accepted flip recomputed before their decision, so the thread count never changes a decision.

## Interpretation and limitations

- Every output is conditional on the final phase and on the fixed inferred pedigree (the released Tier-B edges);
  it is not a calibrated joint uncertainty over pedigree, phase and recombination. Meioses with an unreleased or
  unsampled parent are not observed.
- Posterior expected crossovers, called crossover intervals and observable exposure are separate outputs. A called
  crossover is a supported change of homologue between flanking supported sites; an even number of crossovers
  inside one such interval is not identifiable.
- The input map is a process prior, not ground truth. Informative data can support departures from a positive
  prior, but rates in weakly observed regions remain prior-sensitive, a zero prior forbids crossover mass in that
  interval, and structure entirely between informative sites is allocated by the input map, not learned. Bins
  without exposure have NaN rates, not zero recombination.
- Samples with fewer than two observed parents are decoded in their family MAP frame, which depends on the family
  phase's release state (see [family phase](family_phase.md)).
- Reference-order junctions appear as crossover pile-ups unless a breakpoint file masks them; the junction table
  points to candidates but never changes calls or maps.
