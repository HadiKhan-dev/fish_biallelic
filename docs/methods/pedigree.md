# Pedigree scoring and release

Pedigree inference compares no-, one- and two-observed-parent states using painted ancestry, raw read evidence and genome-wide consistency. Every other sample is a candidate parent; names supply no biological prior. N is samples, C contigs, B bins, M selected markers, S source states, T admitted parent pairs and A configuration rows. Tempering, composite scores, projected source bridges, the bounded M2 panel and loopy family corrections are explicit approximations. Resampling supports and released edges remain distinct from complete best guesses.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="ps01"></a>
## ps01 Scoring-side candidate universe, dyad sets and evidence-free candidates

Treat every sample as a target and every other sample as a candidate parent. Availability is component-specific: it requires an observed painting site whose likelihood spread exceeds the named threshold. An unavailable parent reduces M1 to M0, or M2 to the other parent’s raw M1; its exposure is zero. Availability, information-exponent counts and excursion support have different informative tests and must remain separate.

Implementation: [stages/pedigree/prepare.py](../../haplotype_reconstruction/stages/pedigree/prepare.py), [science/pedigree/prepare.py](../../haplotype_reconstruction/science/pedigree/prepare.py), [science/pedigree/scoring.py](../../haplotype_reconstruction/science/pedigree/scoring.py).

Parameters by name: `INFORMATIVE_GL_SPREAD`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps02"></a>
## ps02 Marker selection and component admission

Select markers within half-open painting bins, keeping all in a small bin or evenly spaced truncated-index positions otherwise. Require every painted position to exist in evidence. Admit a component when selected likelihood triples are nonuniform, without substituting the observation mask for that test. Preserve painting component order and gaps in component indices. Anchoring is determined before subsampling; background and unanchored track statuses become unknown classes.

Implementation: [science/pedigree/prepare.py](../../haplotype_reconstruction/science/pedigree/prepare.py).

Parameters by name: `markers_per_bin`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(L); O(N M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps03"></a>
## ps03 Information exponents of the tempered view

Group bins across the contig in prepared-component order, starting a new information group only after the preceding running count reaches the marker budget. Count every empty bin as one for grouping. Per child, count nonidentical likelihood triples in the selected markers and form the capped information exponent; no informative markers gives zero. Split the exponents back into component arrays. Sample batches retain the complete chromosome-wide bin/marker axis, so groups still cross component boundaries; only one batch of marker GL is concatenated at a time. This tempering approximates correlated information.

Implementation: [science/pedigree/prepare.py](../../haplotype_reconstruction/science/pedigree/prepare.py).

Parameters by name: `markers_per_information_block`, `effective_markers_per_information_block`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N ΣM). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps04"></a>
## ps04 Selector switch probabilities and measured map lengths

Use the run map’s interval distance, including fallback tails, for Haldane selector-switch probabilities and apply the selector floor. The shared Haldane helper supplies the expm1 form. Separately compute measured interval lengths from the measured-coordinate view: clip to supplied knots unless the fallback rate is explicitly measured, and use zero measured support without either. Model transition distance is not measured exposure.


Implementation: [science/pedigree/prepare.py](../../haplotype_reconstruction/science/pedigree/prepare.py), [science/pedigree/scoring.py](../../haplotype_reconstruction/science/pedigree/scoring.py).

Parameters by name: `SELECTOR_SWITCH_FLOOR`, `rate_is_measured`, `unit_morgans`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps05"></a>
## ps05 Exact source chain: factors and posterior marginals

Use painting’s tempered emissions and recorded label transitions for the ordered-diplotype source chain. Contract the transition through its Hamming structure, normalising backward right weights at each boundary. Symmetrise the initial law only; evidence-free samples use a uniform initial law and unit right weights. Available samples losing supported mass raise an invariant error. Preparation fills the dense factor output in sample order. It unpacks cached emissions or gathers raw GL for one sample batch, runs the same independent candidate recurrences, then releases the batch before proceeding. No all-sample emission array, bin-major transpose or whole-component raw GL projection is retained. Error sample indices remain on the original contig axis. Save pair masses before projection consumes the marginal workspace.

Implementation: [stages/pedigree/prepare.py](../../haplotype_reconstruction/stages/pedigree/prepare.py), [science/pedigree/sources.py](../../haplotype_reconstruction/science/pedigree/sources.py), [science/pedigree/prepare.py](../../haplotype_reconstruction/science/pedigree/prepare.py).

Parameters by name: `robustness_epsilon`, `emission_log_floor`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N B S²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps06"></a>
## ps06 Transmitted-state projection: selected-homologue propagation and transmitted ALT

Start from the symmetrised source marginal and propagate selected-homologue orientation with direct/swapped flows. Normalise the propagated law and flows together; lost mass raises. Named states transmit their state ALT value unchanged, including fallback values at uncalled sites. Read-condition only a fractional BACKGROUND frequency, retaining the unconditioned value if its denominator is nonpositive. Computing transmitted ALT costs O(N M S), not a pair-state loop.

Implementation: [science/pedigree/projection.py](../../haplotype_reconstruction/science/pedigree/projection.py).

Parameters by name: `robustness_epsilon`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N M S) for transmitted ALT, plus O(N B S²) posterior propagation; no extra state axis in the ALT calculation.

<a id="ps07"></a>
## ps07 Maximum-entropy bridges

Validate and clip rounding-sized negative residual flows; a zero residual uses the diagonal branch and skips later checks. Normalise both flow vectors by the row total, validate Hall loads, and choose the Hall state by load, marginal mass and explicit index convention. Use the star branch at the Hall boundary, otherwise bracket the all-small or one-large root. Bisect to the named cap or adjacent endpoints, preferring the lower endpoint on equal residual. Choose the pivot, construct arms, then check nonnegativity and reproduced marginals. The bridge tolerance is a post-check tolerance, not a root tolerance; failure has no fallback.

Implementation: [science/pedigree/projection.py](../../haplotype_reconstruction/science/pedigree/projection.py).

Parameters by name: `bridge_max_iterations`, `HALL_ULPS`, `HUB_MASS_TIE`, `release`, `tolerance`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N B (S² + iterations S)); iterations is capped by the named bridge bisection limit.

<a id="ps08"></a>
## ps08 Source-gap excursion mixture

Compute excursion strength from the smaller adjacent-bin informative-marker count and the support target; a nonpositive support setting disables it. Apply the excursion after each bridge only when its strength is positive. Mix the bridged values with a uniform share calculated from the pre-bridge input mass along that axis. The mixture represents uncertain source identity through evidence gaps.

Implementation: [science/pedigree/projection.py](../../haplotype_reconstruction/science/pedigree/projection.py), [science/pedigree/forward.py](../../haplotype_reconstruction/science/pedigree/forward.py).

Parameters by name: `source_gap_support_sites`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N M); O(S²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps09"></a>
## ps09 Null source of an unobserved parent slot

Build a child-specific null frequency by subtracting its available contribution from the total candidate mass, clipping rounding residues and adding the positive symmetric prior. Derive diagonal/off-diagonal transition coefficients from the source transition and selector probability. A genuinely negative persistence raises; a rounding-sized negative persistence saturates to the uniform kernel. Execute the declared coefficient form in axis order. The null slot is unobserved parentage, not a named individual.

Implementation: [science/pedigree/forward.py](../../haplotype_reconstruction/science/pedigree/forward.py).

Parameters by name: `null_prior_samples`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N B S). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps10"></a>
## ps10 Child emission and composite forward score of M0, M1 and M2

Initialise M0, M1 and M2 from null or observed-parent source laws. At each boundary update the first slot then the second, bridge before excursion. Multiply observed-marker emissions in site order; recompute a whole unstable bin in log space. Emission normalization uses correctly rounded division followed by the power on every probability/cache route, skipping the power at unit exponent. A zero exponent still runs normalisation and can have a rounding-sized increment. Pool full and tempered log scores at the fixed composite weight; this is not an independent-marker likelihood.


Implementation: [science/pedigree/forward.py](../../haplotype_reconstruction/science/pedigree/forward.py), [science/pedigree/selective_m1.py](../../haplotype_reconstruction/science/pedigree/selective_m1.py).

Parameters by name: `mismatch_probability`, `COMMON_EMISSION_CACHE_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O((M_obs + B)·S²); O(S); O(N (M + B) S²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps11"></a>
## ps11 Block-certificate upper bound on chromosome M1

Bound M1 with called alleles exact and uncertain alleles relaxed to endpoint likelihoods. Group available candidates by painted words in two block partitions, choosing a content-hash reference member, and apply outward numerical padding. Take the minimum partition bound and sum components outward; unavailable candidates get exact M0. Bounds choose which exact cells are queried, but never replace a retained exact score. Validity must hold for unqueried cells too.

Implementation: [stages/pedigree/screen.py](../../haplotype_reconstruction/stages/pedigree/screen.py), [science/pedigree/selective_bounds.py](../../haplotype_reconstruction/science/pedigree/selective_bounds.py), [science/pedigree/certificates.py](../../haplotype_reconstruction/science/pedigree/certificates.py).

Parameters by name: `log_floor`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(batch B S² m); O(batch·G·B·S²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps12"></a>
## ps12 Certified M1 search, genome retention and the floor profile (CompactM1)

Query the complete maximal-bound tie class first, then exhaust the best frontier and floor frontier in that order. Compute retention with the dimension-fixed contiguous chromosome reduction before completing every local maximum. Apply the resulting floor profile exactly once to nonretained eligible identities. Every query verifies finite exact scores below their bounds; every unknown local cell must lie strictly below the completed local maximum. Batch width and cache lifecycle cannot change retention or the floor.

Implementation: [science/pedigree/selective_screen.py](../../haplotype_reconstruction/science/pedigree/selective_screen.py), [science/pedigree/selective_m1.py](../../haplotype_reconstruction/science/pedigree/selective_m1.py).

Parameters by name: `LOG_FLOOR`, `markers_per_information_block`, `information_tempering_power`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(C·batch·N); O(C ΣR). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps13"></a>
## ps13 Robust screen utilities and search diagnostics

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Compute robust diagnostic utility from tied-rank evidence and a centred softmax of floored chromosome scores with contamination and chromosome weights. The softmax is not clipped at the score floor. Rank utilities only to describe the already bounded panel, after count release and before the guard. A panel is not a utility prefix, and these utilities decide no call.

Implementation: [science/pedigree/panel.py](../../haplotype_reconstruction/science/pedigree/panel.py).

Parameters by name: `rank_weight`, `chromosome_contamination`, `maximum_contig_weight_ratio`, `information_tempering_power`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(C Σ_c n_c); O(C N²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps14"></a>
## ps14 M2 pair bound by state aggregation

Aggregate high-mass and BACKGROUND states separately, placing other named states in a single low group. Build upper transition factors and child emission envelopes, with partner caps only on that parent’s BACKGROUND axis. Contract the second parent axis before the first, preserving this bound’s own arithmetic order. Retry unstable absorption in log space, sum components and round stored float32 bounds upward. Check group counts before narrowing their dtype. Bound values determine panel membership and must be reproduced, not merely remain valid.

Implementation: [science/pedigree/pair_bound.py](../../haplotype_reconstruction/science/pedigree/pair_bound.py).

Parameters by name: `pair_bound_high_mass`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(R² B G²); O(G³); O(K³). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps15"></a>
## ps15 Bounded M2 panel admission

For the full fit and each leave-one-contig-out fit, compare every retained-parent pair bound with the best known M0/raw-M1 score. Admit a pair if any fit reaches the inclusive relevance margin or has a NaN gap, then apply pair eligibility and canonical parent ordering. Weighted sums skip zero weights and add positive rows in contig order. The bounded M2 panel has no top-parent cap; it is a score-defined model approximation. A per-child pair cap (`max_pairs_per_child`) bounds it. A child with more admitted pairs keeps the best by its largest gap over the fits, with the whole tie class at the last kept gap and every NaN-gap pair. It is flagged (`capped_children`, `M2PanelCapped`): its M2 evidence then covers an incomplete panel. Relevance rejection certifies that a pair is outside the competitive margin in every screened fit. The cap has no such guarantee: it can discard a competitive pair. Low depth can admit quadratically many pairs in the eligible-parent count, which motivates the cap.

Implementation: [stages/pedigree/screen.py](../../haplotype_reconstruction/stages/pedigree/screen.py), [science/pedigree/panel.py](../../haplotype_reconstruction/science/pedigree/panel.py).

Parameters by name: `pair_relevance_margin`, `max_pairs_per_child`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(C² P); O(C P). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps16"></a>
## ps16 Panel M2 scoring, fallback rows and the floored chromosome record

For each component, obtain pair masses before projection, score M0, score unique fallback dyads for pairs with exactly one available parent, then score M2 where both are available. Use raw fallback M1 or M0 for unavailable slots. Sum in component order. Publish raw zero/two scores and the already floored one-parent profile; the one-parent field is not raw and must not be floored again.

Implementation: [stages/pedigree/scores.py](../../haplotype_reconstruction/stages/pedigree/scores.py), [science/pedigree/scoring.py](../../haplotype_reconstruction/science/pedigree/scoring.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(T (M + B) S²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps17"></a>
## ps17 Posterior-expected exposure (bins and measured map length)

Compute anchored shared/explained and fully anchored nonmatching pair probabilities, clipping rounding residues. Each unordered edge is evaluated with its lower sample index in the child role and then mirrored. Integrate the minimum adjacent-bin exposure over capped measured intervals. Evidence-free members contribute zero; diagonal matched/exposed values are retained, while diagonal measured length is zero. Exposure is posterior expectation, not an observed meiosis count.

Implementation: [science/pedigree/exposure.py](../../haplotype_reconstruction/science/pedigree/exposure.py).

Parameters by name: `unit_morgans`, `report_edge_coverage`, `report_pair_explainability`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O((E/2 + T)·B·S(S+1)/2); O((E + T) B S²/2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps18"></a>
## ps18 Minimum painted junction counts and callable bins

Pool locally identical selected-marker allele trajectories by first occurrence and run the exact minimum-switch dynamic programme over unordered trajectory pairs. Unknown labels permit every compatible trajectory. Return the minimum final change count and count nonnegative labels on both tracks as callable bins. The parsimonious count does not identify a unique event placement.

Implementation: [science/pedigree/junctions.py](../../haplotype_reconstruction/science/pedigree/junctions.py).

Parameters by name: `junction_unreachable_state`, `junction_unreachable_count`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N B J²); O(J²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps19"></a>
## ps19 Minimum-switch schedules and joint transmission likelihood

Canonicalise selected-marker trajectory content and pool locally identical classes. Trace each sample's
minimum-switch unordered trajectory pair per bin, taking the first minimum path in pair order or the last under
the explicit reversal convention. Alternative equally parsimonious whole paths are not integrated. Build the
schedule, its local homologue classes, the mask of bins with both painted labels known and the number T of distinct
whole-component trajectories together. Count summaries for graph ordering come from that same trace.
[PD28](#pd28) rebuilds these schedules for relationship evidence.

The direction evidence is the joint likelihood of the two samples' scheduled homologues, defined in
[PD06](#finite-continuous-direction). Reset its four-state phase law independently at each component.

Implementation: [stages/pedigree/scores.py](../../haplotype_reconstruction/stages/pedigree/scores.py),
[science/pedigree/junctions.py](../../haplotype_reconstruction/science/pedigree/junctions.py),
[science/pedigree/transmission.py](../../haplotype_reconstruction/science/pedigree/transmission.py),
[science/pedigree/scoring.py](../../haplotype_reconstruction/science/pedigree/scoring.py).

Parameters by name: `mismatch_probability`, `transmission_switch_floor`. Values and rationale are in the
[generated parameter reference](../reference/parameters.md). The founder-mosaic junction probability is derived
from theta and T, rather than set by an additional parameter.

Cost and scope: schedule construction O(N B T²) time and O(B T²) traceback workspace per active sample;
likelihood O(D B) time with four states and constant workspace per dyad. Stored direction inputs take O(N B).

<a id="ps20"></a>
## ps20 Full-marker exclusion e-value

At jointly observed sites compare opposing homozygote evidence with the compatible-genotype replacement bound. Average products over the fixed bets and deduplicated whole-contig/half-overlapping windows, then average over configured contigs, including those without informative evidence. Pattern encoding and direct evaluation must agree. Release uses the genome lower endpoint; an interval straddling the threshold raises instead of guessing.

Implementation: [stages/pedigree/exclusion.py](../../haplotype_reconstruction/stages/pedigree/exclusion.py), [science/pedigree/exclusion.py](../../haplotype_reconstruction/science/pedigree/exclusion.py).

Parameters by name: `alpha`, `genotype_replacement_probability`, `bet_sizes`, `window_widths_bp`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(D L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps21"></a>
## ps21 Completed-window exclusion certificate

Visit sites in order, saving starting-window prefixes before evaluating completed windows in stable stop order. Subtract the declared numerical guards and certify only when the completed-window lower bound proves the genome threshold. Otherwise finish and reduce bets outside windows in the reference order. Store exact values as point intervals and certified values with an unbounded upper endpoint. Publish exclusion only when count-release target dyads exist.

Implementation: [science/pedigree/exclusion_bound.py](../../haplotype_reconstruction/science/pedigree/exclusion_bound.py), [science/pedigree/run.py](../../haplotype_reconstruction/science/pedigree/run.py).

Parameters by name: `alpha`, `genotype_replacement_probability`, `bet_sizes`, `window_widths_bp`, `exclusion_pattern_limit`, `exclusion_dyad_batch`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: At most the full bet/window dyad scan; certification can stop a dyad after a completed window proves the threshold.

<a id="ps22"></a>
## ps22 Scoring orchestration, records, identities and resume

Publish prepared, selective-screen, pair-bound, floored-score and exclusion records under their distinct checked identities. Reuse the selective screen as a whole, other products at their declared contig granularity, and rebuild the deterministic candidate panel from records. Keep state counters distinct from timing diagnostics. Worker count, completion order, resident caching and memory pacing cannot alter scientific values or admission.

Implementation: [stages/pedigree/exclusion.py](../../haplotype_reconstruction/stages/pedigree/exclusion.py), [stages/pedigree/prepare.py](../../haplotype_reconstruction/stages/pedigree/prepare.py), [science/pedigree/run.py](../../haplotype_reconstruction/science/pedigree/run.py), [stages/pedigree/records.py](../../haplotype_reconstruction/stages/pedigree/records.py), [contracts/pedigree.py](../../haplotype_reconstruction/contracts/pedigree.py).

Parameters by name: `preparation`, `scoring`, `exclusion`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Task/record orchestration plus the costs of prepared components, bounds, scores and exclusion; no extra search.

<a id="pd01"></a>
## pd01 Candidate universe (every sequenced sample)

Construct the unrestricted candidate universe: every sample is a child target and every other sample an eligible parent. Use the full eligible multiplicities for state priors, independent of the bounded pair panel. Require unique sample IDs and sufficient sample count. Cohort, sex, generation and metadata labels supply neither eligibility nor parentage truth.

Implementation: [contracts/pedigree.py](../../haplotype_reconstruction/contracts/pedigree.py), [science/pedigree/api.py](../../haplotype_reconstruction/science/pedigree/api.py), [science/pedigree/eligibility.py](../../haplotype_reconstruction/science/pedigree/eligibility.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd02"></a>
## pd02 Decision input assembly: eligible-edge stacking and configuration rows

Decode contig score records one at a time, copying eligible edges in child-major/ascending-parent order. Require identical junction-dyad order across contigs. For each child enumerate M0, ascending M1 identities and canonical panel M2 pairs in that order. Keep full and scored state multiplicities separate. Indexing arrays are C-contiguous; use the composite log likelihoods as supplied.


Implementation: [science/pedigree/stacking.py](../../haplotype_reconstruction/science/pedigree/stacking.py), [stages/pedigree/records.py](../../haplotype_reconstruction/stages/pedigree/records.py), [contracts/pedigree.py](../../haplotype_reconstruction/contracts/pedigree.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(C (N² + T)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd04"></a>
## pd04 Weighted genome likelihood with one M0 contamination mixture

Aggregate chromosome likelihoods with an exact-rounded, order-independent sum of the float64 weighted values. Every row is multiplied, including zero weights, preserving IEEE nonfinite propagation; the separate panel-admission fold omits zero weights. The exact sum retains low contributions independently of array layout or workers. Apply the M0 contamination mixture once to M1/M2 aggregates; leave M0 unchanged. This robust composite transform does not make chromosomes independent observations.


Implementation: [science/pedigree/selection.py](../../haplotype_reconstruction/science/pedigree/selection.py).

Parameters by name: `m0_contamination_probability`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(C A). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd05"></a>
## pd05 Exposure gate and map units

Require every named edge and M2 pair to meet inclusive exposed-bin, exposure-fraction and map-unit thresholds, with no tie band. Count each represented contig once for map-unit support even when bootstrap weight repeats it. Nonfinite measured length contributes the declared unit fallback. An untestable row retains M0 as its state score and has no finite identity score. Validate edge membership and required exposure fields.

Implementation: [science/pedigree/selection.py](../../haplotype_reconstruction/science/pedigree/selection.py).

Parameters by name: `minimum_edge_exposed_bins`, `minimum_pair_exposed_bins`, `minimum_exposed_fraction`, `minimum_exposed_units`, `unit_morgans`, `rate_is_measured`, `preparation`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(C (E + T)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="finite-continuous-direction"></a>
<a id="pd06"></a>
## pd06 Joint-transmission orientation evidence and the direction correction

For an ordered dyad (child c, parent p), score the joint likelihood of their four scheduled homologues.
One homologue of c is a meiotic copy of p's two scheduled homologues. The hidden state is (h, s): which child
homologue is copied, and which parental homologue the meiotic selector carries. At the start of each component
these four states have independent uniform probability 1/4. Direction-common founder starting factors are omitted.
At a bin boundary, the selector stays with probability 1 − theta and switches with probability theta, using the
prepared meiotic selector probabilities (numerically bounded by `transmission_switch_floor` and 1/2).

At an informative bin the emission is 1 − delta if the copied child's and selected parent's local painted classes
agree, and delta otherwise, where delta = `DirectionParameters.mismatch_probability` = 0.01. If either sample
has an unknown painted label in that bin, the copy emission is 1: the bin supplies no class comparison. Scheduled
continuity and mosaic transition terms remain in the calculation through those bins.

The direction leaves three homologues unexplained by copying: p's two and c's other one. Each is a founder mosaic.
At a boundary its trajectory stays with probability 1 − theta; a specified change to another trajectory has
probability theta / (T − 1), derived by dividing the change mass uniformly over the other T − 1 distinct
whole-component trajectories. This is not a free junction-rate constant. When T = 1, no trajectory change is
possible. At each boundary, preserve only homologue alignments that minimise that sample's switches, averaging
straight and crossed alignments equally when tied. Apply this independently to parent and child. Sum over the
four-state paths, rescaling at each bin and accumulating natural log mass, and add independent component log
likelihoods. The stored quantity is `transmission_log_likelihood`, in nats.

For each chromosome compare the forward and reverse joint log likelihoods. With d = log L(c ← p) − log L(p ← c)
and epsilon = `chromosome_error_probability`, its robust orientation log odds are

`log[(1 − epsilon) sigmoid(d) + epsilon/2] − log[(1 − epsilon) sigmoid(−d) + epsilon/2]`.

Require positive bins and callability. Jointly impossible or exactly equal directions contribute zero. Sum in
contig order with resampling multiplicities, then apply the existing pair-failure direction correction to finite
configuration scores. Reciprocal messages, resampling, release tiers retain their
existing definitions; the relationship guard is defined in [PD23](#pd23). The junction-density `order_probability` in [PD07](#pd07) still supplies graph search order
and reports; it is a separate quantity. This intermediate direction evidence differs from final released
orientation log odds.

**Identifiability limit.** A parent-offspring pair sharing one unrecombined homologue on every chromosome is
near-neutral under this evidence. Alternative mismatch paths can leave residual direction odds of a fraction of a
nat rather than exactly zero. The shared homologue itself does not identify direction, and the trio likelihood
decides these cases. A child homologue that recombines the parent's two can supply direction. Painted junction
totals do not identify biological ages in an inferred founder frame.

Validation uses constructed schedules, independent Python path sums, homologue-order symmetries, saved likelihood
and decision replays, and known-truth simulations. Inspect lost, gained and reversed true edges separately: net
edge counts can hide losses. Direction log odds determine pedigree calls, release tiers and edges, and can affect
subsequent phase and recombination outputs.

Implementation: [science/pedigree/transmission.py](../../haplotype_reconstruction/science/pedigree/transmission.py),
[science/pedigree/orientation.py](../../haplotype_reconstruction/science/pedigree/orientation.py).

Parameters by name: `mismatch_probability`, `transmission_switch_floor`, `chromosome_error_probability`,
`pair_failure_probability`. Values and rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(D B) for the likelihood and O(C U) for the chromosome contrasts; the likelihood has four states,
independent of founder count once the schedules are built.

<a id="pd07"></a>
## pd07 Junction-density order probability (graph visit order, role probability)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Compute a weighted junction-density contrast over contigs with common callability, including missingness variance. Insufficient total weight leaves a neutral order probability; otherwise use the weighted sum, not its mean, in the normal approximation. This retained heuristic supplies the graph visit order’s primary key and can therefore affect which cyclic child is displaced.

Implementation: [science/pedigree/orientation.py](../../haplotype_reconstruction/science/pedigree/orientation.py).

Parameters by name: `order_minimum_weight`, `order_variance_offset`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(C N²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd08"></a>
## pd08 Preparatory reciprocal cavity messages (fixed passes)

Starting from zero, execute the fixed number of synchronous reciprocal cavity passes after adding direction evidence. Add incoming slot terms before equal-state log multiplicity, retaining the specified first-maximum and exclusion scans. Children with no finite row send a neutral avoidance message. These preparatory messages build the ancestry factor only and are discarded afterwards; there is no convergence test here.

Implementation: [science/pedigree/family.py](../../haplotype_reconstruction/science/pedigree/family.py).

Parameters by name: `message_passes`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(4 (A + N²)); O(N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd09"></a>
## pd09 Short-ancestry path factor (bounded panels)

Build a bounded panel of finite configurations per sample by strict replacement of the first minimum slot. If a boundary tie overfills the panel, move the entire boundary class to the tail. Compute short-ancestry avoidance, excluding rows that name the child and counting each permitted ancestor once in slot order. Add the resulting nonpositive term once and freeze it before the reciprocal solve. The tail and bounded panel approximate longer ancestry dependence.

Implementation: [science/pedigree/family.py](../../haplotype_reconstruction/science/pedigree/family.py).

Parameters by name: `ancestry_path_budget`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(A R); O(N² R); O(T R²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd10"></a>
## pd10 Converged final reciprocal solve and release orientation log odds

Run the undamped reciprocal solve from zero, returning newly computed messages when residual meets tolerance. At the declared progress checks, pause an unpromising trajectory; if needed restart damped from zero, then resume the paused undamped state within its remaining budget. Preserve the NaN residual refold and accumulation order. Full-fit nonconvergence is fatal; resampled nonconvergence abstains while retaining its draws. Apply final incoming messages once and form release orientation from cavity differences.

Implementation: [science/pedigree/family.py](../../haplotype_reconstruction/science/pedigree/family.py).

Parameters by name: `final_max_iterations`, `final_tolerance`, `final_progress_window`, `final_pause_budgets`, `retry_iterations`, `retry_damping`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(I (A + N²)); O(N + widest child rows). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd11"></a>
## pd11 Tie-aware winners, tie tolerance and the full-fit tie census

Scan rows for the first strict finite maximum and runner-up. Abstain when their margin is within the magnitude-aware winner tolerance, including exact ties; never use row index to resolve a scientific winner. Rank-probability groups instead anchor at their first value. The explicit recorder captures only the full-fit/state/informative census; diagnostics never enter resampled selection.


Implementation: [science/pedigree/ties.py](../../haplotype_reconstruction/science/pedigree/ties.py).

Parameters by name: `contrast_ulp_factor`, `noise_floor`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd12"></a>
## pd12 Marginal state, then identity

Marginalise configuration scores by state using full eligible multiplicities and configured state priors. First require a unique finite parent-count state, then a unique finite identity within that state. Preserve separate unresolved-state and unresolved-identity reasons. Record the compared score scale for graph conflict handling, then pass local choices to acyclic selection.

Implementation: [science/pedigree/selection.py](../../haplotype_reconstruction/science/pedigree/selection.py).

Parameters by name: `state_priors`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(A). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd13"></a>
## pd13 Acyclic graph selection

Keep local rows immediately if they form a DAG. Otherwise peel cyclic components, removing all tied weakest children as conflicts and uniquely weakest children for reconsideration. Run greedy selection on the original local rows excluding only tie conflicts, in role/confidence/convention order. Prefer the local row when feasible; otherwise seek unique feasible winners from its state downward. Iterate the same coordinate order to the pass limit and report exhaustion. This bounded acyclic repair is not global graph optimisation.

Implementation: [science/pedigree/graph.py](../../haplotype_reconstruction/science/pedigree/graph.py).

Parameters by name: `graph_local_search_passes`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N + edges); O(N·(N + edges) + A). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd14"></a>
## pd14 M0 prior sensitivity

Reselect the full adjusted scores under each declared M0 sensitivity prior, reusing the nominal prior result when equal. Do not recompute family-message terms. Require agreement for an M0 state and its exact graph configuration at release; a sensitivity veto has explicit status precedence. Resampled fits do not repeat this prior analysis.

Implementation: [science/pedigree/decision.py](../../haplotype_reconstruction/science/pedigree/decision.py).

Parameters by name: `prior_sensitivity`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(A). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd15"></a>
## pd15 Informative chromosomes and informative map units

For each chromosome, apply the abstaining winner rule to its own parent-count state evidence. A chromosome is informative when any state has a unique winner; it need not agree with the genome-wide state or configuration. Sum the declared informative map units, using the nonfinite-length fallback. Apply the later release threshold without a tie band. Without measured support the units are contig counts, not inferred physical map length.

Implementation: [science/pedigree/decision.py](../../haplotype_reconstruction/science/pedigree/decision.py).

Parameters by name: `minimum_informative_units`, `unit_morgans`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(C A); O(C E). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd16"></a>
## pd16 Chromosome bootstrap and LOCO resampling

Draw all chromosome bootstrap multiplicities first from the named fixed stream, then coalesce identical vectors in lexicographic order. Schedule leave-one-contig-out fits first and bootstrap fits by the declared stable sparsity order. Accumulate integer selection counts with each vector’s multiplicity. A family-message nonconvergence makes the entire fit abstain but leaves its multiplicity in the denominator; other failures stop the stage. Scheduling cannot change support fractions. The stage groups consecutive planned fits into batches by estimated alternative-row scans, with a work ceiling and a target of two waves per planned worker when the fit count allows. Each process attaches the read-only inputs once per batch and evaluates its fits serially. The gather restores the complete original plan before reducing integer counts; batching neither redraws multiplicities nor changes denominators.

Implementation: [stages/pedigree/execution.py](../../haplotype_reconstruction/stages/pedigree/execution.py), [science/pedigree/resampling.py](../../haplotype_reconstruction/science/pedigree/resampling.py).

Parameters by name: `bootstrap_replicates`, `bootstrap_seed`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(C A + C N² (PD07) + (4 + I)(A + N²) + R A + R N² + R² T + graph). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd17"></a>
## pd17 Tier reach and chromosome-count precheck

Compute the maximum achievable bootstrap and leave-one-contig-out support from distinct represented contigs and their map units. Derive each tier’s minimum contig count from both support requirements rather than hard-coding it. Refuse too little informative evidence and report unreachable tiers. Reach is an evidence-availability statement, not a released pedigree call.

Implementation: [science/pedigree/run.py](../../haplotype_reconstruction/science/pedigree/run.py), [science/pedigree/resampling.py](../../haplotype_reconstruction/science/pedigree/resampling.py).

Parameters by name: `minimum_exposed_units`, `unit_morgans`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(R C); O(C²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd18"></a>
## pd18 Tier A/B state and exact-configuration release

Test state support and informative units first, including M0 prior agreement. Require at least one identified named parent for a non-M0 state. Test parent stability and orientation separately; exact configurations additionally require local/graph agreement, configuration support and prior agreement. All fraction thresholds are inclusive. Preserve status precedence for prior veto, no stable parent, insufficient state support, graph conflict, orientation and identity ambiguity.

Implementation: [science/pedigree/release.py](../../haplotype_reconstruction/science/pedigree/release.py).

Parameters by name: `tier_b`, `tier_a`, `minimum_informative_units`, `orientation_reversal_probability`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd19"></a>
## pd19 Complete (full-data acyclic) view

Publish the complete acyclic best-guess view from graph rows, retaining explicit tie-conflict, displacement and unresolved-state reasons. A displaced row is not tier eligible merely because it is acyclic. The complete view is not a release and remains unchanged by the sibling guard.

Implementation: [science/pedigree/release.py](../../haplotype_reconstruction/science/pedigree/release.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd20"></a>
## pd20 Marginal (partial) edges

Start with exact released configurations. For other children without graph conflict and with sufficient informative units, retain a slot only when local and graph parents agree and both sets of parent supports pass. Require resolved orientation to release the edge, otherwise record orientation withholding. The state itself need not have passed the tier. Increase the minimum parent count from retained edges without lowering its unresolved maximum.

Implementation: [science/pedigree/release.py](../../haplotype_reconstruction/science/pedigree/release.py).

Parameters by name: `minimum_informative_units`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd21"></a>
## pd21 Tier-B candidate sets

For a resolved Tier-B state, cover the finite configuration probability mass in named tie-class order, completing the cut class. Wide classes retain value ordering; ordinary classes use the declared content convention. Convert configurations to sides only when their structure permits it and name singleton sides through parent support and orientation. Apply marginal-edge and count-release overrides to typed views. Candidate sets retain ambiguity and are not extra released edges.

Implementation: [science/pedigree/release.py](../../haplotype_reconstruction/science/pedigree/release.py).

Parameters by name: `support_set_coverage`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(rows log rows); O(A log A). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd22"></a>
## pd22 Full-marker count release (boundary unit; file outside area list)

Score every candidate of unresolved/count-only Tier-B targets and reject threshold-straddling exclusion intervals. Remove compatible reverse directions already supported by nonwithheld outgoing edges. Release a count only with complete evidence, sufficient units, no graph conflict, at most one survivor, agreement among survivors/supported/complete parents and no guard-withheld dependency. Keep resampling fractions unchanged and label the distinct exclusion-based release evidence.

Implementation: [science/pedigree/count.py](../../haplotype_reconstruction/science/pedigree/count.py), [science/pedigree/run.py](../../haplotype_reconstruction/science/pedigree/run.py), [science/pedigree/exclusion.py](../../haplotype_reconstruction/science/pedigree/exclusion.py).

Parameters by name: `alpha`, `minimum_informative_units`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(targets × N); O(D L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd23"></a>
## pd23 Sibling-as-parent guard (fixed point and application)

Test every pre-guard Tier-B edge p → c using the painting relationship evidence in [PD28](#pd28).
The guard has no parameter group of its own; its numerical settings are the relationship parameters in PD28.
Use every member of c's relatedness family except c and p, removing a member linked by a released parent
edge to c or another member. Relatedness support sets the witness's prior price; it does not filter membership.
There is no minimum family size or different-parent cleaning rule.

For each sequence let d be its FEASIBLE log likelihood minus the largest of its four process log likelihoods,
including FEASIBLE itself, after summing each process over configured contigs. The pair margin is m = d(c,p).
For witness w, d_par = d(w,p) and d_sib = d(c,p,w). Its full-sib prior pi is `sibship_prior` at the band of
the smaller support of c and w, bounded by `support_bounds`; unknown support uses the lowest band.
The price is c_w = max(0, log(pi/(1−pi))). First check each witness against the family itself: with at least one
other witness, d_fam(w) is the median of d(c,w,w′) over its current check partners. The child's ring is all other
relatedness-family members in sorted identifier order, cyclically. Select the next `sibship_check_partners` after w
on that fixed ring, then keep only the edge's current witnesses; do not refill the window after filtering. With no
retained partner, the witness is not checked. At most P+2 family members give exactly the all-partner check. For an even count use the mean of
the two middle values. If d_fam(w) <= −c_w, set the witness aside under both hypotheses, contribute zero and count
it in `set_aside`. A single witness has no family check. Otherwise the contribution is
`v_w = max(d_par, −c_w) − max(d_sib, −c_w)`. Either hypothesis can set a witness aside at its prior penalty.
The final score is `Lambda = m + sum(v_w)`, in nats. This sum combines profile likelihood ratios of separate
sequences; witnesses share c and p and are dependent. It has no posterior-odds interpretation. The
`withholding_score` of log(1000) nats is calibrated on known-truth designs including three-quarter siblings.

Withhold below `−withholding_score`. A retained edge has `family` evidence above the positive score threshold,
and `pair_only` evidence otherwise. An edge without witnesses and with zero pair margin is `untested`;
a nonzero pair margin is tested even without a family. A prior at most one half has price zero and contributes no evidence, giving an effective support floor of 0.5.
A single witness cannot withhold while the pair margin is above
`−(withholding_score − top price)`, where the top price is log(0.98/0.02). Two top-band witnesses can still withhold. The sibling-guard table reports final witnesses, set-aside/against/supporting counts,
pair/family/total score, final evidence call, irreversible call, withdrawal iteration and pair/family basis.
A witness is against below −1 nat and supporting above +1 nat. Set-aside counts include a contradicted family
check or both hypothesis margins at or below minus its price. Withholding basis is `pair` when the pair margin
alone crosses the negative threshold, otherwise
`family`; a retained edge has `none`.

Recompute all original edges against the retained edge set until no more withdrawals, in at most E+1 passes.
Withdrawals are irreversible; a prior withdrawal can remain an abstention under final premises, recorded by its
separate evidence call and first-withdrawal basis. Freeze requests for every original family member, including
linked members that can return after a premise is withdrawn. Request dyads (w,p), trios (c,p,w) and the bounded
ring trios (c,w,w′), in addition to the focal pair (c,p), for every original family member other than c and p.
Trio requests are unordered and deduplicated across every role. There are at most (f−1)P ring trios per child and
O(f²P) sequences per family.
Accumulate witnesses in sorted identifier order. Sum process log likelihoods in component order, then contigs
in configured order, before taking any maximum. All withdrawals in a pass use one snapshot of retained edges. Score each requested dyad and trio once.
Run count release with the withheld set, re-slot, then verify identical edge and withheld sets. Filter withdrawn
edges without renumbering remaining slots. Guarded views clear calls while retaining candidate lists; complete and
diagnostic views stay pre-guard. The four released relationship tables gain `relationship_basis`; a row with
multiple released parents is `family` only if every one has family evidence, otherwise `pair_only`, and is empty
when no parent is released. Existing guard-withheld status texts are retained.

Implementation: [science/pedigree/guard.py](../../haplotype_reconstruction/science/pedigree/guard.py).

Parameters by name: `withholding_score`, `sibship_prior`, `support_bounds`, `sibship_check_partners`. Values and rationale are in the
[generated parameter reference](../reference/parameters.md).

Cost and scope: linked cleaning uses retained-edge adjacency; ring checks take O(E · f · P) per pass.
Chromosome evidence is scored once, including members that can return in later passes.

<a id="pd24"></a>
## pd24 Released parent order and ParentEdges

Rank released parent pairs by descending observed-site counts in configured-contig order, reading only until all pairs are distinguished, then by sample name. Apply the explicit slot convention; a single parent starts in the first slot, while later guard filtering preserves a surviving second slot. Order edges child-major then slot. Bind both the exact conditioning digest and the product derivation identity. Slots carry no sex or biological parent-order meaning.

Implementation: [science/pedigree/api.py](../../haplotype_reconstruction/science/pedigree/api.py), [science/pedigree/slots.py](../../haplotype_reconstruction/science/pedigree/slots.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N log N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd25"></a>
## pd25 Report tables and published views

Construct typed report views after the declared release, slot and guard steps. Distinguish the final Tier-B count-consistent state call from pre-release state support. Keep diagnostic alternatives in internal row order, exact exclusion values separate from interval endpoints, and unresolved scalar values missing. Format tuples, names and ParentSlotBasis only in outputs. Reports cannot feed a new decision.

`painted_junctions_per_morgan` is the sample's minimum painted founder switches, summed over scored chromosomes, divided by twice its callable painted map length. For chromosome c, callable length is `working_morgans[c] * callable_bins[c,n] / (2 * total_bins[c])`; the rate is per homologue per Morgan of the configured working map. Working length sums the Haldane distances `-0.5 * log(1 - 2 * theta)` of the actual selector transitions over scored components and bin boundaries (PS04). It includes the configured fallback rate even when that rate is not declared measured, and is distinct from the measured exposure used by pedigree gates. Zero callable length gives NaN. The number is reported only and never enters a gate, tier or parent-call decision.

Values of a few per Morgan reflect the crossovers of the cross's generations: supported simulation designs give values from 0 to 4. Values far above that, such as 25–46 in a real cohort with many unrelated founders, indicate fragmented paintings; parent calls based on those paintings are not reliable. These ranges describe interpretation, not decision thresholds. The rate depends on the configured map and scales inversely with its length: doubling a configured recombination rate doubles the denominator and halves this number. An erroneous configured rate therefore rescales the diagnostic.

Implementation: [outputs/reports/pedigree.py](../../haplotype_reconstruction/outputs/reports/pedigree.py), [science/pedigree/views.py](../../haplotype_reconstruction/science/pedigree/views.py).

Parameters by name: `report_edge_coverage`, `report_pair_explainability`, `support_set_coverage`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(A + N·columns); O(A log A). The report-only junction rate adds O(C N) arithmetic and O(N) temporary storage. These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd26"></a>
## pd26 Relative-proxy contrasts

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Evaluate relative-proxy contrasts for the specified comparison configurations and publish them as diagnostics. Preserve pre-guard view labels, including edges subsequently withheld. No release tier, parent edge or count decision reads this table; a favourable proxy contrast is not a pedigree assertion.

Implementation: [stages/pedigree/relatives.py](../../haplotype_reconstruction/stages/pedigree/relatives.py), [science/pedigree/relatives.py](../../haplotype_reconstruction/science/pedigree/relatives.py).

Parameters by name: `mismatch_probability`, `scoring`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B · S² · m); O(K²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd27"></a>
## pd27 Stage orchestration, records, identities and consistency checks

Validate samples and informative-contig reach, then prepare, screen, bound and score before full decision/resampling. Build count-exclusion and relationship requests, score them per contig, solve the guard, perform count release, orient slots and construct the typed product. Recheck edge/withheld consistency and reject a count resting on a withheld direction. Publish only under the checked identity; execution and formatting stay outside the pure decision steps.

Implementation: [contracts/pedigree.py](../../haplotype_reconstruction/contracts/pedigree.py), [science/pedigree/run.py](../../haplotype_reconstruction/science/pedigree/run.py), [stages/pedigree/decision.py](../../haplotype_reconstruction/stages/pedigree/decision.py), [stages/pedigree/records.py](../../haplotype_reconstruction/stages/pedigree/records.py), [stages/graph.py](../../haplotype_reconstruction/stages/graph.py), [interface/run.py](../../haplotype_reconstruction/interface/run.py).

Parameters by name: `reverse_conventions`, `preparation`, `scoring`, `panel`, `exposure`, `direction`, `selection`, `resampling`, `release`, `exclusion`, `relationship`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Sum of preparation, scoring, decision, resampling, exclusion and report-view work; independent contig/fit tasks are placed by key.

<a id="pd28"></a>
## pd28 Painting relationship observations, processes and chance laws

Rebuild the minimum-switch schedules exactly as [PS19](#ps19). Each sample has two bin-local classes and a mask
requiring both painted labels. Class numbers compare samples within a bin only. The pair observation is zero
when either class of one sample matches either class of the other, and one otherwise. The trio observation is zero
if two latent parents, each carrying at most two classes with repeats allowed, can explain all three unordered
genotypes as full siblings; otherwise it is one. More than four distinct classes among the six is infeasible.
An observation needing an unknown label is skipped.

The model is defined on the grid of known bins. The chance chain conditions on the previous known observation
of that component, and the state chain uses the summed Haldane distance between consecutive known bins. This is
the observation-grid model rather than a marginalisation of latent emissions at unknown bins.

The distance of prepared selector boundary theta is `−0.5 log(1−2 theta)` Morgans. Distances add through skipped
bins; there is no extra switch floor in this evidence. Each component starts independently in the stationary law
of a continuous-time S/N process. With R = lambda_in + lambda_out and pi_N = lambda_in/R, a known-bin gap g gives
`P(S→N) = pi_N (1−exp(−R g))` and `P(N→S) = (1−pi_N) (1−exp(−R g))`.
S emits one with probability `isolated_error * (1−q_t)` and zero with the complementary probability; an isolated
painting error can show infeasibility only where the chance law permits it. N emits a cohort chance chain with first-bin P(0)=q_t,
`P(1|previous 1)=s`, and `P(1|previous 0)=min(1−1e−6,(1−q_t)(1−s)/q_t)`.
Use a scaled forward sum and accumulate natural log likelihoods. The four processes, in stored order, are
FEASIBLE (`failure_region_rate`, `1/failure_region_length`), FS-like (4/3,4), D2-like (2,2), and NEVER (always N).
Rates are per Morgan; the failure length is in Morgans. FS-like and D2-like rates match the occupancy and mean
flux of the biological processes, not their tract-length laws. With bin-specific q_t the chance chain is not
stationary at q_t; q_t supplies its first-bin law and the specified conditional transition. Independent components and contigs contribute additively
before the process comparison in [PD23](#pd23).

Estimate chance laws separately for each component from the entire cohort. Clip both persistence estimates to
[0.01,0.99]. If f_a is the frequency of class a
among homologues of samples with both labels known, the pair chance law is
`q_t = 1 − sum_ab f_a f_b (1−f_a−[a!=b]f_b)^2`, clipped to [1e−9,1−1e−9].
This pair chance law describes two independent genotypes drawn from the cohort class frequencies. In 15 of 16
validation cases it is 0.002–0.046 below the cohort's empirical pair sharing. The trio chance law is the empirical
feasibility of relationship-blind cohort triples.
Pair persistence pools adjacent-bin 1→1 events over all unordered sample pairs, without a requested-pair filter.
Unknown bins break persistence event adjacency; component boundaries never supply events. The no-event
persistence is 0.5. A bin with no known homologues uses the upper clipped chance probability.

Draw `reference_triples` triples of distinct samples per component without regard to relationships. Draw on the
sample identifiers in sorted order using the named `pedigree.relationship.reference` PCG64 stream seeded by the
pedigree bootstrap seed plus the zero-based configured contig index and the semantic component index as the
stream key. Each component has its own stream; completion order cannot select draws.
Identifiers serve as keys only. Trio phi_t is the fraction of known references with observation one;
`q_t = clip(1−phi_t, 1e−3, 1−1e−3)`. With no known references phi_t is zero. Pool adjacent 1→1 events over the
reference triples for trio persistence, again using 0.5 without events. Store the references and both chance laws
with each component's evidence. Per-contig records bind preparation, parameters, request, axes and random seed;
completed contigs resume independently and are published under an owned relationship key. Requested observations
are scored in fixed chunks of `relationship_sequence_chunk` sequences (execution grain only). Binary sequence
workspace is O(chunk · B), with at most two such arrays during likelihood conversion; likelihood outputs remain
O(requests · 4). The all-pairs persistence estimator stores two integer vectors of length N and does not allocate
an N²-by-B observation array.

**Limits.** The pedigree stage cannot assign parents in a cross with many unrelated founders and few offspring per founder. It works from reconstructed founder haplotypes, which require many offspring per founder. In such a cross it leaves every sample without an assigned parent, including samples whose parents are sequenced.

This score combines profile likelihood ratios on scheduled paintings with calibrated witness priors.
Witnesses are dependent, and the score is not a posterior probability. Bin resolution, painting errors and cohort
representation affect the evidence. Feasibility alone
does not establish direction or parenthood. Relatives masked by shared founder classes without informative
sibship evidence can remain released as `pair_only`; the model reports that limit. Direction,
resampling, release tiers and the genotype exclusion evidence for count release retain their own definitions.

Implementation: [science/pedigree/relationship.py](../../haplotype_reconstruction/science/pedigree/relationship.py),
[stages/pedigree/relationship.py](../../haplotype_reconstruction/stages/pedigree/relationship.py).

Parameters by name: `failure_region_rate`, `failure_region_length`, `isolated_error`, `withholding_score`,
`sibship_prior`, `support_bounds`, `reference_triples`, `sibship_check_partners`, `relationship_sequence_chunk`,
`bootstrap_seed`. Values, units and rationale are in the
[generated parameter reference](../reference/parameters.md).

Cost and scope: O(B) per requested sequence and process; O(E · f · P) requested work before deduplication,
including ring sibship checks, with at most O(f²P) distinct sequences per family. Pair persistence uses
O(N² B) and reference trios O(K_ref B) per component. Sequence kernels parallelise over independent sequences.

<!-- named mechanism anchors -->

## Named mechanisms

<a id="acyclic-graph-selection"></a>

**Acyclic graph selection:** see [PD13](#pd13).

<a id="candidate-panel"></a>

**Candidate panel:** see [PS15](#ps15).

<a id="child-forward-model-and-composite-score"></a>

**Child forward model and composite score:** see [PS10](#ps10).

<a id="chromosome-resampling"></a>

**Chromosome resampling:** see [PD16](#pd16).

<a id="configuration-likelihoods-and-the-exposure-gate"></a>

**Configuration likelihoods and the exposure gate:** see [PD04](#pd04).

<a id="converged-final-reciprocal-solve"></a>

**Converged final reciprocal solve:** see [PD10](#pd10).


<a id="exposure-and-junction-counts"></a>

**Exposure and junction counts:** see [PS17](#ps17).

<a id="full-marker-mendelian-exclusion-of-m0-and-m1-counts"></a>

**Full marker mendelian exclusion of m0 and m1 counts:** see [PD22](#pd22).

<a id="inputs-and-outputs"></a>

**Inputs and outputs:** see [PD02](#pd02).

<a id="marginal-state-then-identity"></a>

**Marginal state then identity:** see [PD12](#pd12).

<a id="one-weighted-fit"></a>

**One weighted fit:** see [PD04](#pd04).

<a id="parallel-structure"></a>

**Parallel structure:** see [PD27](#pd27).

<a id="parent-candidates"></a>

**Parent candidates:** see [PS01](#ps01).

<a id="preparation-markers-source-factors-and-information-blocks"></a>

**Preparation markers source factors and information blocks:** see [PS02](#ps02).

<a id="products-and-published-tables"></a>

**Products and published tables:** see [PD25](#pd25).

<a id="published-tables"></a>

**Published tables:** see [PD25](#pd25).

<a id="ragged-quadratic-scoring-of-m0-m1-and-m2"></a>

**Ragged quadratic scoring of m0 m1 and m2:** see [PS16](#ps16).

<a id="relative-proxy-contrasts-of-released-edges"></a>

**Relative proxy contrasts of released edges:** see [PD26](#pd26).

<a id="release-tiers"></a>

**Release tiers:** see [PD18](#pd18).

<a id="released-parent-order-release-22"></a>

**Released parent order:** see [PD24](#pd24).

<a id="sibling-as-parent-guard"></a>

**Sibling as parent guard:** see [PD23](#pd23).

<a id="transmitted-state-projection"></a>

**Transmitted state projection:** see [PS06](#ps06).
