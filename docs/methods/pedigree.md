# Pedigree scoring and release

Pedigree inference compares no-, one- and two-observed-parent states using painted ancestry, raw read evidence and genome-wide consistency. Every other sample is a candidate parent; names supply no biological prior. N is samples, C contigs, B bins, M selected markers, S source states, T admitted parent pairs and A configuration rows. Tempering, composite scores, projected source bridges, the bounded M2 panel and loopy family corrections are explicit approximations. Resampling supports and released edges remain distinct from complete best guesses.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="material-units"></a>
## Material units and regimes

Pedigree preparation, map-scale frames and projected evidence read painting/refit_reference: the reference-resolution painting decoded again on the founder refit's founders (stage repaint, which runs in the pedigree step after every contig's refit). The order is painting -> founder refit -> second painting -> pedigree; the refit keeps the original painting as its input and never reads the second one. The second painting's record identity binds the reference painting's identity (hence the final assembly, the evidence, the working map, the run calibration and the fine painting) and the refit's. Both resolutions share the fine label-continuity outcome and the estimated contig genotype-outlier mass; reference-resolution copying chains are fitted separately. All pedigree bin constants keep their reference-bin units.

The pedigree decides by the amount of material, never by a count of chromosomes, contigs or components.
The genome axis is the working-map coordinate of every scored painting bin: contigs in configured order, painting
components in map order. For a component's selector probabilities theta, each boundary adds the Haldane distance
`-0.5 * log(1 - 2 * min(theta, 0.499999))`; the component's first bin sits at the running offset of the components
before it, so gaps between components (and half-bins at their ends) count nothing. U, the run's material, is the
last coordinate: the working Morgans of its informative components. The configured map is used whether or not its
rate is declared measured; a wrong rate scales U, and with it every cut below, by the same factor. Painting
fragmentation can move U: a gap between painted components counts nothing.

The axis is cut into consecutive units of 1 working Morgan (`MaterialParameters.unit_morgans`). A final piece shorter
than 0.5 Morgan (`tail_morgans`) joins the unit before it, so a run holds round(U) nominal units. A bin belongs to
unit min(floor(x), last); a bin exactly at a cut belongs to the unit on its right; empty nominal units are omitted.
Units may straddle components and contigs: chromosomes assort independently, so a straddling unit is no more
dependent on its neighbours than a unit inside one chromosome. Unit composition depends on the configured contig
order, a deterministic convention.

The regime reads U alone and is chosen before score tasks, uniformly for every child:

- U < `assessable_morgans` (5): a completed `not_assessable` pedigree: unresolved states, count bounds [0, 2], no
  edges or counts, no best guesses and absent fit diagnostics.
- `assessable_morgans` <= U < `standard_morgans` (12): low material. Tier B only, with configuration and parent
  bootstrap thresholds `low_bootstrap` (0.95) and leave-one-unit-out (LOBO) fraction `low_leave_one_out` (1.0);
  exposure and informative minimums of one unit.
- U >= `standard_morgans`: the released Tier B thresholds (0.70 / 0.70 / LOBO 0.80) with minimums of three units.
  Tier A (0.95 / 0.95 / 0.95) is assessed only when also U >= `tier_a_morgans` (12, so from the same 12 Morgans);
  below it Tier A is not published and does not order parent slots.

Every unit counts one in exposure, the informative minimums, the bootstrap and LOBO, whatever its length. The cuts
are an empirical calibration on known-truth simulations, not variance-matched resampling: units of one chromosome
are correlated at lag one (adjacent 1-Morgan units 0.12 to 0.22 for relatives, lag two at most 0.04), so one
10-Morgan chromosome in ten units holds about 7.3 to 8.2 effective units. The calibration simulated 60 genomes of
single chromosomes from 0.5 to 15 Morgans (six cohort designs: complete families, small families, mixed, backcross,
random half-sibs and a parent-free sib-rich design; depth 1 and 5) with the configured map equal to the generating
map, fitted both threshold sets on every whole-chromosome layout and compared each U band with the released
thresholds on multi-chromosome genomes of at least 15 Morgans:

- Below 5 Morgans the low-material rule released non-parents in mixed-family designs (12 of 971 releases at 3.5 to
  5 Morgans, none of 600 at 5 to 7) and in small families; from 5 Morgans every design passes.
- Below 12 Morgans the released thresholds released more non-parents than at 15 Morgans and above in small families
  (about 2 % against 1.45 %) and more edges in the parent-free sib-rich design (0.1 to 0.5 per 100 parentless
  children against 0.03); from 12 Morgans no design shows an excess. The tests have little power per design near
  the cut (0.1 to 0.4): these are the cuts the data support, not demonstrated equivalence.
- In designs with sequenced parents the low-material rule released no non-parent in 74,328 releases from 8
  Morgans, at 0.5 to 3 points lower recall than the released thresholds.

Limitations of the rule:

- **The cuts are in true Morgans and the rule reads the working map.** U is measured on the map the run uses (the
  map file, else `rate_cm_per_mb`), whether or not the rate is the organism's. A map k times too long scales every
  cut down by k in true Morgans: with a configured rate 2.5 times the true one, 12 configured Morgans are about
  4.8 true Morgans, where the released thresholds release relatives as parents (a sib-rich parent-free design read
  on such a map with the cut at 9.5 released 4.5 edges per 100 parentless children). The cuts hold only on a
  working map in true Morgans: a measured map or a correct genome-wide rate. Since 4.19 the painting-based
  estimate of the fallback rate guards the regime ([map scale](#map-scale)): by default a call is released only when
  the configured map's and the estimated map's regimes both release it.
- **Sib-rich parent-free designs.** Where no parent is sequenced and full-sib families are large, both tiers can
  release a sibling or other relative as a parent at any material level: in the calibration Tier A released 2, 2
  and 1 such edges at 12-13, 13-15 and at least 15 Morgans, so no higher Tier A cut removes them. An end-to-end
  check released one Tier A non-parent in 448 parentless children of such a design at 13.7 Morgans.
- **Rare events above the cut.** The end-to-end check (64 cells: 10 to 15 Morgans on one or two chromosomes, and
  12 Morgans on two chromosomes plus 24 scaffolds) found two single events above 12 Morgans for which the
  chromosome-axis rule gives no reference, because it assessed no Tier A in those cells and made no wrong
  zero-parent call: the Tier A non-parent above, and one wrong zero-parent call in a small-family design at depth 1
  (13 Morgans), within the released thresholds' small-family rate (1 in 946 at 15 Morgans and above). A parent-free
  design released one Tier B non-parent (1.7 edges per 100 parentless children); elsewhere non-parent releases
  equalled or fell below the chromosome-axis rule's on the same cells.
- Long single chromosomes and sib-rich parent-free designs remain the least tested inputs; unit composition and the
  orientation cap read the same working map.

Each contig's score task scores its pieces of the genome units (a piece is a maximal run of one component's bins in
one unit) from the barrier's cut list, computed from every informative prepared layout; the score identity binds the
contig's offset, last unit and cut list, and the screen identity binds every informative contig's preparation, so a
change in an earlier contig re-scores the later ones. Forward log increments are credited per bin without restarting
chains; junctions use the component's traced minimum-switch schedule; transmission keeps its whole-component
four-state filter; so every piece field is a sum of per-bin increments and the stacking sums pieces into units
exactly. Retained M1 uses raw piece scores. Other identities share the child's floor
`M0_unit + bins_unit / bins_genome * (sum_k F_k - M0_genome)`: the screen's genome floor deficit spread over all units
by bins, so the floors of a child sum to its genome floor. This is an imputed profile, not raw M1. Per-contig
reports (evidence summary, painted junctions) are built from the pieces before they are summed into units.

The M2 panel holds every retained eligible pair of a child below the per-child cap, so the full fit, the unit
omissions and the bootstrap fits all see the same complete set of two-parent configurations of retained parents; a
capped child's panel is incomplete in every fit and flagged. Resampling is conditional on full-data
preparations, source factors and schedules: later increments still depend on the filter states propagated through
an omitted unit. It measures conditional stability, not calibrated confidence or independent segregation.

LOBO aggregation is linear in the unit count: full additive totals minus one unit, distinct presence counts minus
that unit's flags, and graph-order moment totals minus its moments; the decisions equal per-unit refits. Each fit
still refits family and graph selection. Arrays cost H (units) times their eligible-edge/panel axes; each float64 H by
E array costs 8 H E bytes, plus N by N orientation statistics, shared inputs, score decode/stack buffers and worker
scratch. At N = 2,800 the decision task needs about 1.1 GB per unit (about 47 GB for a 42-Morgan genome) and the
score records about 0.19 GB per unit.

Parameters by name: `unit_morgans`, `tail_morgans`, `switch_clip`, `assessable_morgans`, `standard_morgans`,
`tier_a_morgans`, `low_bootstrap`, `low_leave_one_out`. Values and provenance are in the
[generated parameter reference](../reference/parameters.md).

Implementation: [science/pedigree/regimes.py](../../haplotype_reconstruction/science/pedigree/regimes.py),
[science/pedigree/blocks.py](../../haplotype_reconstruction/science/pedigree/blocks.py),
[science/pedigree/stacking.py](../../haplotype_reconstruction/science/pedigree/stacking.py),
[science/shared/material.py](../../haplotype_reconstruction/science/shared/material.py).

<a id="map-scale"></a>
## Map scale: the material regime under an estimated fallback rate

The cuts above are in true Morgans, but U is measured on the working map. Where the working map is the fallback rate
(`[genetic_map] rate_cm_per_mb`: contigs without a map file, and stretches beyond a map file's knots), the pedigree
checks that rate against a painting-based estimate before it chooses the regime. The estimate is the late of the
run's two rate estimates ([rl20](relatedness.md#rl20) is the early one): it never changes the working map of any
stage, only which regime's rule the pedigree applies.

**Estimate (er).** One genome-wide rate r in cM/Mb, from the released paintings of the relatedness families of at
least two members. For each member (the target) and painting component, the other carriers (members and the
family's sequenced slot samples, firm stage-1 tracks only) define a frame wherever they show four parental classes
with four distinct bin keys and exactly one supported 2 + 2 split; consecutive frames are linked by shared classes
into the two parents' homologue chains (a link across an undetermined stretch needs both classes shared). The frame's
emission is the painting's own pair emission of the target for the four full-sib configurations (power 1, at the path
strength, ratios to their maximum), mixed with a misfit column (the mean over all state pairs) at a weight delta fitted
by maximum likelihood on the 4-state chain. The estimate maximises the composite likelihood of every frame sequence
under an error-run chain: the hidden pair (true transmission state, shown state), the true state following the Haldane
chain of the target's two meioses at r per Morgan of physical distance x 1e-8, error runs starting with probability
eps and ending with probability xi per frame (profiled; eps = 0 is the plain chain). With the painting's read state
every one of these chains carries the target's read class (below). A relationship guard first removes
members whose evidence is not full-sib like: a pair test (the full-sib IBD-count chain must beat the half-sib and
parent-offspring chains by `guard_nats`, each at its own best rate; a member is kept when it passes with at least half
of the others) and an iterative target test (tracks that may leave the parents' classes; a removal needs a gain above
`guard_nats` and at least `target_share` posterior non-full-sib frames; the parental switches follow the same Haldane
process in both models, so the test does not select on a sib's crossover count).

- **Read state (`read_classes` > 1; LR phase 2, Tier 3).** Long reads cover many selected sites, so the scalar bin
  emissions count the same reads at every site they cover. Each hidden state then also holds the target's read class
  c, in the transmission frame: homologue 1 carries parent 0's transmitted copy, homologue 2 parent 1's (a physical
  homologue keeps its parent through every crossover, so c carries across frames unpermuted). A frame's emission is a
  transfer matrix of c per emitted state, W_b(x, y) = exp(min(e_b - m, 300) + l_b - t) M~_b(x, y) for the ordered
  pair (x on homologue 1, y on homologue 2): e_b the tempered scalar emission, M_b with log scale l_b the bin's read
  matrix of the painting at the path strength (its builder, the one the source chain uses; the class at the previous
  bin's last selected site carried through the bin's sites, each site's read factor tempered by the bin's weight
  1 / DE), M~ its classes permuted by the homologue swap J when x > y, m the configurations' best scalar emission
  (today's 300-nat overflow guard acts on the scalar ratio) and t one common scale for every column of the frame.
  The misfit matrix is the mean over all unordered pairs, each in both homologue orientations at 1/2 ((M + J M J) /
  2): a pair drawn from the scalar column's law that carries no parent assignment, the class crossing it by the law's
  average transfer. The guard's target chain uses the same matrices per track-symbol pair (track 1 on homologue 1; the
  'other' symbol averages the labels outside the four, an outside pair drawn with replacement; zero when every label
  is one of the four). Between frames the class crosses the undetermined bins by the label-free product of the
  painting's depth-conditioned class operators of their sites (their alleles are not read, as without read state),
  folded once into the next frame's matrices; a sequence starts from the class law at the end of the preceding bin
  (the component's first site: its own law) with uniform transmission states, so a sequence cut at file-map knots
  starts from the marginal law too (pi_(l-1) T_l = pi_l). One normalisation per frame over (state, class). The pair
  test reads painted classes only, no emission. Without read state, and for every sample that bypasses it (deep at
  every site, or a library without detected read sharing), the matrices are 1 x 1 and the chains are the scalar
  chains exactly (C = 1 of the same kernels). The frames' matrices are rebuilt with the read builder for the
  requested bins in each sample batch. One graph task builds each contig's frames. A target removal changes
  the carriers in its own family. The following guard pass reuses every unchanged family's complete sequences
  and rebuilds the affected families on each contig. The fit consumes the final ordered sequences.
- **What it estimates.** The event-weighted mean recombination rate of the meioses of the sibships that give frames.
  In a multi-generation cross those may be one generation's meioses (in a real cichlid cross 175 of 181 expected
  events came from two sibships). This is a statement about which sibships give frames, not about parentage.
- **Bias, stated honestly (review3 MUST-FIX 1).** On held-out simulations the adopted estimates read 0.85 to 1.08 of
  the generating rate. The bias depends on the rate: about -5 % at 5 to 8 cM/Mb, but approximately unbiased at
  2 cM/Mb, where seven genomes read 0.95 to 1.08 (mean +0.2 %; up to +8 %). Rerunning those seven genomes with a
  correct configured map (2 instead of 5 cM/Mb; same meioses) moved the estimate by -0.4 % on average (-3.2 % to
  +2.7 %): the high readings at 2 cM/Mb are the estimator's own, not a pull towards a too-long configured map. The
  simulations follow Haldane's model; under crossover interference the chain over-reads by about 2 to 5 % (an
  untested bias on real data, in the risky direction). The estimate is therefore not a conservative (low) reading at
  low rates, which is why it may only veto by default and is adopted only under test-agree.
- **Material units.** The working units of replication of the consistency test and the interval are equal-material units: the
  material units of the working map along the frame-bearing span of the map-free contigs (configured order, no gap
  between contigs; `MaterialParameters.unit_morgans` and `tail_morgans`, the pedigree's own), each with the exact
  split of every sequence's forward log-likelihood into per-frame predictive terms summed per unit (their sum is the
  sequence's log-likelihood). No count of chromosomes or contigs decides anything (owner, 13 Nov 2026; rev 5 required
  five contigs, bootstrapped contigs and took contigs as the strata of the dispersion).
- **Family consistency.** z_f = d_f / sqrt(phi_f (v_f + v_rest)), d_f the log difference between a family's own
  estimate and the others', v the profile-curvature variances (both read the families' totals), phi_f the
  family-within-unit deviance per degree of freedom of the other families (floored at 1): in each unit, twice the
  gain of one rate per family over one common rate, with one degree of freedom per family with frames there less
  one. z is referred to t(df) at a Bonferroni level `consistency_alpha`; the most deviant family is dropped above
  `drop_margin` z_crit (three or more families) and the test repeats; any remaining |z| > z_crit makes the estimate
  inconsistent. Null calibration (pure families of simulated genomes, 4.21): 1 of 65 development genomes refused,
  the same genome as on contig strata, and 1 of 126 simulated genomes; of the 400 development null families, 5.2 %
  lie beyond the t(df) 0.975 quantile and 1.0 % beyond its 0.995 quantile: no excess false-refusal was observed
  relative to contig strata on these simulations; independence of the units and exact t calibration are not
  established (the t reference is an empirically checked approximation). Against contig strata, units add degrees of freedom (median 81 against 32)
  and detect a shifted family at least as often; no gate decision changed on those genomes.
- **Interval.** A crossed percentile bootstrap (`bootstrap_replicates`, fixed `bootstrap_seed`) of families and
  material units. At least `minimum_interval_units` units and two families are needed.
- **Adopted** (may replace the working map's regime in adopt mode): not at a grid edge, at least `minimum_events`
  expected crossovers on the kept families (2 x frame span x r), 95 % half-width at most `maximum_half_width`, at
  least two families with frames, consistent. **Usable** (may veto): consistent, two families, not at a grid edge, at
  least `usable_minimum_events` expected crossovers and an interval whose lower end lies above the grid floor.

**Map files.** The estimate is fitted on the map-free contigs only: it must estimate the fallback rate, the quantity
it stands in for, and file contigs (another, varying rate) would contaminate it. Frames on map-file contigs, inside
the files' knots, give a diagnostic instead: the same chain on file-map coordinates estimates the ratio of the painted
crossover rate to the file map's, with a warning when its interval excludes 1. A genome fully covered by map files has
the same map under every mode, so the veto never acts there.

**Routes.** `[genetic_map] rate_estimate` (command line `--rate-estimate`):

| Mode | Route | Regime |
|---|---|---|
| `ignore` | working_map | the working map's alone; the estimate and its warnings are reported, nothing reads them (the forced configured map) |
| `veto` (default) | both_maps when the estimate is adopted or usable, else working_map | a call is released only when both maps' regimes release it |
| `adopt` | estimate when adopted and its interval excludes the working rate (test-agree, review3 MUST-FIX 2); both_maps when adopted with an interval covering the working rate, or usable; else working_map | the estimated map's regime alone on the estimate route |

The working rate is the working map's fallback rate: `rate_cm_per_mb`, or in adopt mode the early estimate when
[rl20](relatedness.md#rl20) adopted it. Both estimates are fitted on paintings made under that one working map.

**Material under the estimated map.** U is recomputed from the same prepared layouts: preparation records, for every
selector boundary, the part of its bin-centre interval measured by the fallback rate (all of it on a map-free contig,
the stretches beyond the knots otherwise; `PreparedComponent.selector_fallback_bp`), and each boundary's working
Morgans d becomes d + (r - r_working) 1e-8 x that length, its switch probability recomputed as preparation computes it.
The same is done at the interval's ends: when the cut-offs lie inside [U(lower end), U(upper end)] the run log, the
pedigree summary and the material statement carry a straddle warning, whatever the route (the decision uses the point
estimate).

**Which pedigree quantities read the estimated map.** Only the regime: U against the assessable, standard and Tier A
cut-offs. The material units, the genome axis, unit weights, the D7 floors, the D8 orientation caps and the exclusion
weights keep the working map. The cut-offs are what the calibration fixed in true Morgans and what the map-scale
studies measured; moving the rest would need its own validation and a second scoring of every contig. In adopt mode
the early estimate already moves the whole working map when it is confident.

**Both maps: the explicit intersection.** When the two maps put U in different regimes, the lower regime's rule runs
the whole pedigree and the higher regime's rule runs to its released calls (no publication); a Tier B call is
released only when the higher regime releases the same call (observed-parent count and parents), and an edge only
when both release it. The intersection is computed, not assumed: low-material Tier B is not a subset of standard
Tier B (minimum units 1 against 3, bootstrap cut-offs 0.95 against 0.70; non-nested on 363 of 2,026 calibration
layouts). Withheld calls become unresolved with status `tier_b_material_veto` (release evidence `material_veto`);
edges released by both survive as marginal edges, as after the sibling guard; the views keep the parents before the
veto. A not-assessable lower regime gives the not-assessable product. The route can only withhold relative to the
working map's regime: it never releases a call that regime does not release. When the lower regime is the estimated
map's, its own rule usually withholds what the working map's regime alone would release (the veto pass then has
nothing left to withhold); the report therefore counts what the product withholds relative to the working map alone,
whichever rule withheld it, and the veto pass's share separately. On the estimate route (adopt) the working map's
regime also runs to its released calls, only to be counted (the product can then also add calls).

**Reports.** The run log prints the estimate, its route and its warning once, in every mode, then the material
statement: the regime under the working map alone and under the estimated map, the applied regime, and the Tier B
calls, Tier B edges and Tier A calls the product withholds (and adds) relative to the working map alone.
`pedigree/summary.json` carries `material` (U of each map, `working_regime`, `estimate_regime`, `applied_regime`,
`veto_regime`, the `withheld_*` and `added_*` counts relative to the working map alone, the veto pass's `vetoed_*`
share, straddled cut-offs and the statement; the counts are -1 when the product is not assessable but the working map
alone is, since its releases are then not computed) and, with `veto` and `adopt`, `map_scale` (the estimate, interval,
gate, route, family rows, file-map ratio and warning: record `pedigree/map_scale`; null with `ignore`, where nothing
in the pedigree reads the record).

**Evidence.** On 195 lab genomes the release code reproduces the rev-5 lab estimator's (contig, family) tables to
4e-12 nats and every rev-5 gate decision; the consistency test on material units (4.21) changes none of the release
gate's decisions on them. With the release gate (families x units, no contig rule): adopted estimates
read 0.85 to 1.08 of the generating rate on the development sets (46), held-out sets (23) and single- and
two-chromosome cells (15); none above 1.10; no interval's lower end above the generating rate. Usable-only estimates
(veto) read 0.70 to 1.33; every interval covered the generating rate. The intertwined-families case that rev 5's
contig rule guarded (three contigs, 30 expected crossovers, interval at the grid floor) is neither adopted nor usable.

Cost: frames, guard and fit in seconds to minutes on 8 cores (35 to 225 s on simulated genomes of 66 samples); memory
O(relevant samples x S^2 x B) for the emission caches of every component, and the target test's 25-symbol emissions.
The number of labels enters through the painting's S states. With C read classes, a frame caches 26 C²
values for the guard. The builder computes the matrices for the bins touched by each sample batch and reads
the contig's complete selected-site sequence for the operators between frames. The graph reads paintings on
independent tasks and gathers frame tasks in contig, component and family order. Each target-removal decision
requires the previous round's genome-wide evidence. The rate fit parallelises independent sequence and rate cells.

Parameters by name: `mode`, `guard_nats`, `target_share`, `target_iterations`, `rate_grid`, `coarse_rate_grid_spec`,
`misfit_weights`, `pair_error_rates`, `target_entry_rates`, `target_exit_rates`, `error_run_starts`,
`error_run_ends`, `consistency_alpha`, `drop_margin`, `bootstrap_replicates`, `bootstrap_seed`,
`minimum_interval_units`, `minimum_events`, `maximum_half_width`, `usable_minimum_events`
(`PedigreeParameters.map_scale`). Values and provenance are in the [generated parameter
reference](../reference/parameters.md).

Implementation: [science/pedigree/map_scale.py](../../haplotype_reconstruction/science/pedigree/map_scale.py),
[science/pedigree/map_scale_frames.py](../../haplotype_reconstruction/science/pedigree/map_scale_frames.py),
[science/pedigree/map_scale_kernels.py](../../haplotype_reconstruction/science/pedigree/map_scale_kernels.py),
[science/pedigree/map_scale_chains.py](../../haplotype_reconstruction/science/pedigree/map_scale_chains.py),
[science/pedigree/map_scale_reads.py](../../haplotype_reconstruction/science/pedigree/map_scale_reads.py),
[science/pedigree/material_veto.py](../../haplotype_reconstruction/science/pedigree/material_veto.py),
[science/pedigree/regimes.py](../../haplotype_reconstruction/science/pedigree/regimes.py),
[stages/pedigree/map_scale.py](../../haplotype_reconstruction/stages/pedigree/map_scale.py).

<a id="ps01"></a>
## ps01 Scoring-side candidate universe, dyad sets and evidence-free candidates

Treat every sample as a target and every other sample as a candidate parent. Availability is component-specific: it requires an observed painting site whose likelihood spread exceeds the named threshold. An unavailable parent reduces M1 to M0, or M2 to the other parent’s raw M1; its exposure is zero. Availability, information-exponent counts and excursion support have different informative tests and must remain separate.

Implementation: [stages/pedigree/prepare.py](../../haplotype_reconstruction/stages/pedigree/prepare.py), [science/pedigree/prepare.py](../../haplotype_reconstruction/science/pedigree/prepare.py), [science/pedigree/block_scoring.py](../../haplotype_reconstruction/science/pedigree/block_scoring.py).

Parameters by name: `INFORMATIVE_GL_SPREAD`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps02"></a>
## ps02 Marker selection and component admission

Select markers within half-open painting bins, keeping all in a small bin or evenly spaced truncated-index positions otherwise. Require every painted position to exist in evidence. Admit a component when selected likelihood triples are nonuniform, without substituting the observation mask for that test. Preserve painting component order and gaps in component indices. Anchoring is determined before subsampling; background and unanchored track statuses become unknown classes.

Implementation: [science/pedigree/prepare.py](../../haplotype_reconstruction/science/pedigree/prepare.py).

Parameters by name: `markers_per_bin`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(L); O(N M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps03"></a>
## ps03 One likelihood per configuration

Every observed selected marker of a child enters each configuration's forward chain once, at full strength (4.22 R4 1a): there is no tempered view, information exponent, information group or pooled average of two views, and no transferred working temperature from the painting's confidence divisor. The dependence of markers covered by the same reads stays inside the model through the child's read state ([child read state](#child-read-state)); the parents' source chains keep the painting's path strength ([ps05](#ps05)), a declared design that keeps full sibs apart from parents.

Implementation: [science/pedigree/read_forward.py](../../haplotype_reconstruction/science/pedigree/read_forward.py), [science/pedigree/prepare.py](../../haplotype_reconstruction/science/pedigree/prepare.py).

Parameters by name: none (the information-block, effective-marker and transferred-temperature settings are removed).

Cost and scope: one forward chain per configuration, O((M_obs + B)·S²).

<a id="ps04"></a>
## ps04 Selector switch probabilities and configured material

Use the run map's interval distance, including fallback tails, for Haldane selector-switch probabilities and apply the selector floor. The shared Haldane helper supplies the expm1 form. The selector probabilities, clipped below one half, also give each component's working-map material ([material units](#material-units)). Each contig additionally records its configured material, m(L) - m(0) of the run map (the map and its fallback-rate tails) with L its last evidence position: the contig's fixed weight in the exclusion e-value ([PS20](#ps20)).


Implementation: [science/pedigree/prepare.py](../../haplotype_reconstruction/science/pedigree/prepare.py).

Parameters by name: `SELECTOR_SWITCH_FLOOR`, `switch_clip`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B). These terms describe the unit's work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps05"></a>
## ps05 Source chain with the painting's read state

Every sample as a candidate parent is an HMM over (ordered diplotype u, read class c), S² C states, read at the painting's path strength on its reference grid. From (u, c) at bin b to (v, c') at bin b + 1 the weight is A_b(u, v) E_{b+1}(v)[c, c'], E_b(v)[c, c'] = exp(e_b(v) + l_b(v)) M_b(v)[c, c']: A_b the painting's label-level Hamming transition, e_b the tempered scalar bin emission, M_b the bin's read matrix and l_b its log scale (the class at the previous bin's last selected site carried through the bin's sites to the class at its last site; the first bin starts from the first site's class law). An ordered v with v1 > v2 reads the unordered pair's matrix with both class indices permuted by J (the homologue swap of the classes), so the chain is invariant under (a, b, c) ↔ (b, a, J c). Without read state (`read_classes` = 1, or a sample deep at every site) C = 1 and M = 1: the scalar chain, through the same kernels. The diplotype prior is uniform; the painting's panel-error state is left out.

The read matrix tempers each site's per-class read factor with the bin's weight w, f_l(c) = R_l(c)^w.
The untempered factors average to one over the site's class law, Σ_c π_l(c) R_l(c) = 1. By Jensen's inequality the
tempered average is strictly below one when 0 < w < 1 and R_l varies among classes with positive probability under
π_l. This deficit arises at heterozygous allele configurations, which can also occur for a repeated BACKGROUND
label. Over a bin the class-path average of the tempered product is at most the untempered bin likelihood to the
power w. The source chain uses the bin-level correction (4.22 D5) to remove this deficit.

Each pair's bin matrix keeps its tempered shape (its class dynamics), and its log scale becomes
l_b(v) = w log(α_b M_b^1(v) 1) − log(α_b M̂_b(v) 1), with M_b^1 the untempered bin matrix including its scale and
M̂_b the stored, rescaled tempered matrix. Thus α_b exp(l_b) M̂_b 1 = (α_b M_b^1 1)^w. The total from the entering
class law α_b is the bin's untempered read likelihood to the power w, the same exponent used for the scalar
emission. A second pass over (sample, label pair) starts with the first bin's entry site class law and normalises
α_{b−1} M̂_{b−1}(v) to get α_b. This approximates the source predictor by carrying the class law along the fixed
label pair, omitting label transitions and scalar emissions from the carry. Each pair carries its own α even
when pairs share a bin's matrix. In exact arithmetic the correction multiplier is at least one and equals one
at w = 1, where rounding contributes about 10⁻⁸ nats. The correction makes a bin without read sharing exactly
neutral.

Without the correction, short paired-end reads at low per-sample depth, whose read evidence is weak, lost their
true parents to the penalty. Zero_stress kept 41 of 68 Tier B edges (68 with the correction or without the read
state) and bc8 at 2.5x released none (120 with the correction). Long reads kept their read-state gains, which the
scalar source chain loses entirely (famhs 0 of 98). The painting, its calibration, family phase and the map-scale
frames take the same correction (4.22 D5P, [painting](painting.md), read-state tempering).

Exact forward-backward per sample: backward class vectors β_b(u, c) (one Hamming contraction per class, max-normalised per bin), then the transmission-oriented propagation over (x, y, c), x the transmitted homologue's label: P_{b+1}(x, y, c') = (1 − τ_b) D(x, y, c') + τ_b D(y, x, J c'), D(v, c') = Σ_{u, c} P_b(u, c) A_b(u, v) E_{b+1}(v)[c, c'] β_{b+1}(v, c') / β_b(u, c), from the homologue-symmetrised first-bin posterior, τ_b the meiotic switch probability ([PS04](#ps04)). Its class-free law is the homologue-symmetrised marginal at every bin (the selector is stationary at one half and independent of the parent's data); the transmitted marginal π and the three flows of the transmitted coordinate (same state, leaving, entering) are the class-summed scalar formulas with each class's right weights r_b(v, c) = Σ_{c'} E_{b+1}(v)[c, c'] β_{b+1}(v, c'), and are normalised with the propagated law. Preparation computes and stores, class-free: the unordered pair masses (exposure), π, the flows and BACKGROUND's transmitted ALT probability ([PS06](#ps06)); no class axis is stored. Evidence-free samples keep the uniform chain (no emission, no read state). Available samples losing supported mass raise an invariant error naming the original sample, the component and the boundary. One bounded, contiguous evidence slice supplies the painting builder at a time. In a mixed slice, the source recurrence uses a one-class batch for samples of unshared libraries and samples that bypass the read state at every site. The read-state samples use their carried widths. Both sets of results are restored to the canonical sample order. Map-scale frames use one-class matrices for bypassed rows, with no class transfer in the gaps.

Implementation: [stages/pedigree/prepare.py](../../haplotype_reconstruction/stages/pedigree/prepare.py), [science/pedigree/read_sources.py](../../haplotype_reconstruction/science/pedigree/read_sources.py), [science/pedigree/sources.py](../../haplotype_reconstruction/science/pedigree/sources.py), [science/pedigree/prepare.py](../../haplotype_reconstruction/science/pedigree/prepare.py).

Parameters by name: `robustness_grid`, `emission_log_floor`, `read_classes`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N B S² C²) time, O(n B S² C) per sample batch of n for the backward vectors; stored O(N B S²) (pair masses) + O(N B S) + O(N M). With C = 5 about C² = 25 times the scalar chain's arithmetic.

<a id="ps06"></a>
## ps06 Transmitted-state projection: prepared laws and transmitted ALT

The projection reads the prepared transmitted marginal and flows ([PS05](#ps05)) and builds the bridges ([PS07](#ps07)). Named states transmit their state ALT value unchanged, including BACKGROUND's frequency at uncalled sites. A fractional BACKGROUND frequency is conditioned on the candidate's own reads in preparation: P(a = ALT | x = BACKGROUND) = Σ_y P_b(BACKGROUND, y) Σ_c P(c at the marker's site | BACKGROUND, y) P(a = ALT | BACKGROUND, y, c, the site's reads), the class posterior from the oriented pair's entering class vector carried forward through the bin's selected sites (their class operators and tempered read factors) and β_b carried backward, and the site's configuration evidence (1 − ε) L_a κ_c(a) + ε B untempered (L the genotype likelihoods, κ the class likelihood ratio, B the cohort law). Without read state κ = 1: the mixed genotype likelihoods; the unconditioned value is kept where the denominator is nonpositive.

Implementation: [science/pedigree/read_sources.py](../../haplotype_reconstruction/science/pedigree/read_sources.py), [science/pedigree/projection.py](../../haplotype_reconstruction/science/pedigree/projection.py).

Parameters by name: `robustness_grid`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L S C²) for the read-conditioned BACKGROUND ALT (L the painting's selected sites; O(N M S) without read state), in preparation; the projection itself O(N B S) plus the bridges.

<a id="ps07"></a>
## ps07 Maximum-entropy bridges

Validate and clip rounding-sized negative residual flows; a zero residual uses the diagonal branch and skips later checks. Normalise both flow vectors by the row total, validate Hall loads, and choose the Hall state by load, marginal mass and explicit index convention. Use the star branch at the Hall boundary, otherwise bracket the all-small or one-large root. Bisect to the named cap or adjacent endpoints, preferring the lower endpoint on equal residual. On the all-small branch the state that sets the bracket end q_max (ties as in the large-branch search) has a square-root fold there: when its discriminant factor 1 − qL is at most the square root of machine epsilon, its diagonal product u is taken from the root equation q(1 + Σu) = 1 instead of its own quadratic root, which a root within rounding of q_max would move by about the square root of machine epsilon. Choose the pivot, construct arms, then check nonnegativity and reproduced marginals. The bridge tolerance is a post-check tolerance, not a root tolerance; failure has no fallback.

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
## ps10 Child emission and forward likelihood of M0, M1 and M2

Initialise M0, M1 and M2 from null or observed-parent source laws. At each boundary update the first slot then the second, bridge before excursion. Multiply observed-marker emissions in site order; recompute a whole unstable bin in log space. Each bin's emissions are divided by their largest value before they multiply the forward state, and the normaliser enters the log likelihood. The score is the configuration's one forward log likelihood ([ps03](#ps03)). With the painting's read state the chains also carry the child's read class ([child read state](#child-read-state)).


Implementation: [science/pedigree/forward.py](../../haplotype_reconstruction/science/pedigree/forward.py), [science/pedigree/read_forward.py](../../haplotype_reconstruction/science/pedigree/read_forward.py), [science/pedigree/selective_m1.py](../../haplotype_reconstruction/science/pedigree/selective_m1.py).

Parameters by name: `mismatch_probability`, `COMMON_EMISSION_CACHE_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O((M_obs + B)·S²); O(S); O(N (M + B) S²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="child-read-state"></a>
## Child read state in the configuration chains

With the painting's read state (`read_classes` > 1) every configuration chain of a child (M0, M1, M2 and the second-degree and full-sib proxies) carries the child's read class c (which homologue the covering reads come from) besides its transmitted founder pair (f, s): S² C states. The founder pair moves at bin edges by the bridges and the null source, the same for every class; the class moves between the child's observed selected markers by the painting's class operators composed over the selected sites between them (the read state is defined on the evidence grid), whatever the founders do, and the chain starts from the class law at the component's first selected marker. At an observed marker the emission of the slots' transmitted ALT probabilities (a, b) (slot 1 = homologue 1) is E_c(a, b) = G(a, b) R_c(a, b): G the scalar site likelihood ([PS10](#ps10)) and R_c the GL-anchored read ratio of the painting's factor at full strength, R_c(a, b) = Σ_g W_g(a, b) L′_g R_c(g) / Σ_g W_g(a, b) L′_g, R_c(g) = [(1 − ε) L_g κ_c(g) + ε B] / [(1 − ε) L_g + ε B] (W the product law of the ordered configuration g, L′ the mixed and L the raw genotype likelihoods, κ the class likelihood ratio, B the cohort law). Σ_c π(c) R_c(g) = 1, so without read sharing (operator rows equal to the next marker's class law) the chain is the scalar chain. The chain applies E itself at every observed marker ([ps03](#ps03)). Preparation stores each child's operators into its observed markers and its four configuration ratios per class (float32) and the first marker's class law (ChildReads); a sample deep at every site (the painting's bypass) stores none and keeps the scalar chain exactly, as does every child with one read class, through the same kernels.

When the stored transition into the child's next observed marker has identical rows, its destination law is
independent of the incoming class. For each founder pair, summing the incoming class mass before that transition
gives the same forward likelihood. The recursion can propagate a single value per founder pair across preceding
bin edges, including empty bins, up to that marker. Its class law is restored from the common row for the
marker emission. At the end of the observations only total mass is needed. Tests use exact row equality on the
stored operator. An arbitrarily small difference between rows keeps the full class dimension. The relative proxies
apply the same rule within each inheritance layer.
The pair bound retains its entrywise group maxima over founders before summing destination classes.

The pair bound carries the class too. Each site factor is nonnegative and bilinear in (a, b), so over an allele box (the pair bound's BACKGROUND range) every class entry is largest at a corner: r̄_c = max_corner E_c / max_corner G, and the bin's class transfer bound Π_k T_k diag(r̄_k) is entrywise above the chain's; the pair bound aggregates (class, group, group) with each group pair's entrywise maximum (no partner caps on this route). The M1 screen's certificate of a pair is its exact M1 with the read state: every eligible pair of a child batch is scored once and the screen's queries read those scores. A class-carrying group bound cost a third to a half of that exact pass, and the screen queried 85–100 % of the eligible chromosome pairs on every simulated cohort; the screen's record (best, cutoff, retained pairs, chromosome maxima, floors) depends only on exact scores, so the choice of valid certificates changes no value.

Implementation: [science/pedigree/child_reads.py](../../haplotype_reconstruction/science/pedigree/child_reads.py), [science/pedigree/read_forward.py](../../haplotype_reconstruction/science/pedigree/read_forward.py), [science/pedigree/read_bounds.py](../../haplotype_reconstruction/science/pedigree/read_bounds.py), [science/pedigree/relatives.py](../../haplotype_reconstruction/science/pedigree/relatives.py), [science/pedigree/prepare.py](../../haplotype_reconstruction/science/pedigree/prepare.py).

Parameters by name: `read_classes`, `class_transfer_cache_bytes`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: per configuration O(B C S² + K (C² + C) S²) for K observed markers (the class mix of every state pair, then its factor), against O(B S² + K S²) for one class; the M1 screen one exact chain per eligible pair; the pair bound O(K S² C³) per child to build the bin bounds; stored O(K (C² + 4C)) float32 per component. The chains take each bin's class transfer Π_k T_k diag(f_k) of every state pair no candidate changes from a per-batch cache (`class_transfer_cache_bytes`), computed once per pair of allele words at the bin's observed markers (states with the same alleles there share it); the block scorer advances a tile of one child's rows together bin by bin, so each bin's transfers are read once per tile, and the relatives' proxies read the same cache. The pair bound builds each bin's class bound once per pair of box words. None of this changes a value beyond the order of floating-point operations.

<a id="ps11"></a>
## ps11 Block-certificate upper bound on chromosome M1

Bound M1 with called alleles exact and uncertain alleles relaxed to endpoint likelihoods. Group available candidates by painted words in two block partitions, choosing a content-hash reference member, and apply outward numerical padding. Take the minimum partition bound and sum components outward; unavailable candidates get exact M0. Bounds choose which exact cells are queried, but never replace a retained exact score. Validity must hold for unqueried cells too. With the read state the certificate of a pair is its exact M1: every eligible pair is scored once per child batch and the queries read those scores ([child read state](#child-read-state)).

Implementation: [stages/pedigree/screen.py](../../haplotype_reconstruction/stages/pedigree/screen.py), [science/pedigree/selective_bounds.py](../../haplotype_reconstruction/science/pedigree/selective_bounds.py), [science/pedigree/certificates.py](../../haplotype_reconstruction/science/pedigree/certificates.py).

Parameters by name: `log_floor`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(batch B S² m); O(batch·G·B·S²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps12"></a>
## ps12 Certified M1 search, genome retention and the floor profile (CompactM1)

Query the complete maximal-bound tie class first, then exhaust the best frontier and floor frontier in that order. Compute retention with the dimension-fixed contiguous chromosome reduction before completing every local maximum. Apply the resulting floor profile exactly once to nonretained eligible identities. Every query verifies finite exact scores below their bounds; every unknown local cell must lie strictly below the completed local maximum. Batch width and cache lifecycle cannot change retention or the floor.

The retention band is 60 T nats with T = units ** power, minimum 1, units the chromosome's selected markers with an empty bin counted as one (4.21 B2's effective units at r_T = 0); the genome band uses the summed units. The band and the pair cap ([ps15](#ps15)) are a compute budget of the birth-order decision ([pd08](#pd08)): on every measured root the wide band with the cap and with no cap gave the same decisions (4.22 R4).

Implementation: [science/pedigree/selective_screen.py](../../haplotype_reconstruction/science/pedigree/selective_screen.py), [science/pedigree/selective_m1.py](../../haplotype_reconstruction/science/pedigree/selective_m1.py), [science/pedigree/m1.py](../../haplotype_reconstruction/science/pedigree/m1.py).

Parameters by name: `LOG_FLOOR`, `information_tempering_power`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

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

Aggregate high-mass and BACKGROUND states separately, placing other named states in a single low group. Build upper transition factors and child emission envelopes, with partner caps only on that parent’s BACKGROUND axis. Contract the second parent axis before the first, preserving this bound’s own arithmetic order. Retry unstable absorption in log space, sum components and round stored float32 bounds upward. Check group counts before narrowing their dtype. Bound values determine panel membership and must be reproduced, not merely remain valid. With the read state the aggregated forward carries the child's class and the partner caps are not applied ([child read state](#child-read-state)).

Implementation: [science/pedigree/pair_bound.py](../../haplotype_reconstruction/science/pedigree/pair_bound.py).

Parameters by name: `pair_bound_high_mass`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(R² B G²); O(G³); O(K³). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps15"></a>
## ps15 M2 panel admission

Admit every eligible pair of a child's retained parents (canonical parent ordering); there is no relevance screen. A screen proves exclusion only for the fits it screens, while the decision reuses the panel in every bootstrap and leave-one-unit-out fit: a pair rejected on the full data (or on any fixed family of fits) can be the best configuration of a resampled fit, and its missing row then lowers the two-parent evidence there. Below the cap the M2 evidence is therefore exact over the retained parents in every fit. A per-child pair cap (`max_pairs_per_child`) bounds the panel at cap x N rows. A child with more pairs keeps those with the largest full-data pair bound (the per-contig bounds summed in contig order; no chromosome or unit structure enters), with the whole tie class at the last kept bound and every pair whose bound is not a number. It is flagged (`capped_children`, `M2PanelCapped`): its M2 evidence then covers an incomplete panel, a lower bound, and the cap can discard a competitive pair. Children retain many parents, and so reach the cap, at low depth and in large sib-rich cohorts.

Implementation: [stages/pedigree/screen.py](../../haplotype_reconstruction/stages/pedigree/screen.py), [science/pedigree/panel.py](../../haplotype_reconstruction/science/pedigree/panel.py).

Parameters by name: `max_pairs_per_child`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(C P); O(C P). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps16"></a>
## ps16 Panel M2 scoring, fallback rows and the piece record

For each component, obtain pair masses before projection, score M0, score unique fallback dyads for pairs with exactly one available parent, then score M2 where both are available. Use raw fallback M1 or M0 for unavailable slots. Credit every bin's increment to its piece of the genome's material units ([material units](#material-units)); each contig publishes its pieces, its genome offset and the screen's floor of each child. The stacking sums pieces into units and spreads the genome floor deficit; the one-parent profile is floored there exactly once.

Implementation: [stages/pedigree/scores.py](../../haplotype_reconstruction/stages/pedigree/scores.py), [science/pedigree/block_scoring.py](../../haplotype_reconstruction/science/pedigree/block_scoring.py), [science/pedigree/stacking.py](../../haplotype_reconstruction/science/pedigree/stacking.py).

Parameters: supplied typed inputs and the stage's [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(T (M + B) S²). These terms describe the unit's work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps17"></a>
## ps17 Posterior-expected exposure (bins)

Compute anchored shared/explained and fully anchored nonmatching pair probabilities, clipping rounding residues. Each unordered edge is evaluated with its lower sample index in the child role and then mirrored. Sum over the bins of each piece, so exposure adds over pieces into units. Evidence-free members contribute zero; diagonal matched/exposed values are retained. Exposure is posterior expectation, not an observed meiosis count.

Implementation: [science/pedigree/exposure.py](../../haplotype_reconstruction/science/pedigree/exposure.py).

Parameters by name: `report_edge_coverage`, `report_pair_explainability`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O((E/2 + T)·B·S(S+1)/2); O((E + T) B S²/2). These terms describe the unit's work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps18"></a>
## ps18 Minimum painted junction counts and callable bins

Pool locally identical selected-marker allele trajectories by first occurrence and run the exact minimum-switch dynamic programme over unordered trajectory pairs. Unknown labels permit every compatible trajectory. Return the minimum final change count and count nonnegative labels on both tracks as callable bins. The parsimonious count does not identify a unique event placement.

Implementation: [science/pedigree/junctions.py](../../haplotype_reconstruction/science/pedigree/junctions.py).

Parameters by name: `junction_unreachable_state`, `junction_unreachable_count`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N B J²); O(J²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps19"></a>
## ps19 Minimum-switch schedules and joint transmission likelihood

Order selected-marker trajectories in the contig's canonical REF/ALT coding and pool locally identical classes.
Trace each sample's minimum-switch unordered trajectory pair per bin, taking the first minimum path in pair order
or the last under the explicit reversal convention. Alternative equally parsimonious whole paths are not integrated. Build the
schedule, its local homologue classes, the mask of bins with both painted labels known and the number T of distinct
whole-component trajectories together. Count summaries for graph ordering come from that same trace.
[PD28](#pd28) rebuilds these schedules for relationship evidence.

The direction evidence is the joint likelihood of the two samples' scheduled homologues, defined in
[PD06](#finite-continuous-direction). Reset its four-state phase law independently at each component.

Implementation: [stages/pedigree/scores.py](../../haplotype_reconstruction/stages/pedigree/scores.py),
[science/pedigree/junctions.py](../../haplotype_reconstruction/science/pedigree/junctions.py),
[science/pedigree/transmission.py](../../haplotype_reconstruction/science/pedigree/transmission.py),
[science/pedigree/block_scoring.py](../../haplotype_reconstruction/science/pedigree/block_scoring.py).

Parameters by name: `mismatch_probability`, `transmission_switch_floor`. Values and rationale are in the
[generated parameter reference](../reference/parameters.md). The founder-mosaic junction probability is derived
from theta and T, rather than set by an additional parameter.

Cost and scope: schedule construction O(N B T²) time and O(B T²) traceback workspace per active sample;
likelihood O(D B) time with four states and constant workspace per dyad. Stored direction inputs take O(N B).

<a id="ps20"></a>
## ps20 Full-marker exclusion e-value

At jointly observed sites compare opposing homozygote evidence with the best Mendelian-compatible genotype pair, both scored under the observation model. Average products over the fixed bets and deduplicated whole-contig/half-overlapping windows. The genome e-value is the weighted mean sum_k w_k E_k over the analysed contigs, with w_k their configured material m_k(L_k) - m_k(0) over the total (the run map and its fallback-rate tails; L_k the last evidence position): fixed before any genotype is read, so under the scalar site-independence assumption the mean is an e-value under any dependence between contigs (linearity of conditional expectation). Uninformative contigs keep their weight; a contig without material has weight zero and no task. For certifiable dyads, release uses the genome lower endpoint. An interval straddling the threshold requires exact refinement.

The site ratio (4.22 A11) uses the observation model of [observation.md](observation.md): each sample's genotype likelihoods L(g) are mixed against the site's cohort genotype frequencies q at the sample's own outlier rate e, L'(g) = (1 - e) L(g) + e q·L, the generative reading being that with probability e the cell's genotype is redrawn from q. The ratio is A'/D' with A' = (x'(0) y'(2) + x'(2) y'(0)) / 2 and D' the largest x'(g) y'(h) over Mendelian-compatible pairs (no opposite homozygotes). D' bounds the likelihood of the two cells under every parent-offspring law whose cells are redrawn independently at their rates, and A' is a density of the cells, so the site ratio has expectation at most one under that null. There is no genotype-replacement constant and no floor: the allowances are the samples' own rates, the painting calibration's per-sample point estimates (each sample's grid maximum refined between its grid neighbours by the three-point parabola in log e, clamped to them and only where concave). A calibration with a single candidate rate estimates no per-sample rates; every sample then takes the run's rate. No error control is claimed for products over dependent sites beyond the scalar site-independence assumption above. On identical inputs every site ratio is at least (1 - e_x)(1 - e_y) times the 4.21 ratio, whose allowance max(0.01, e_x + e_y) bounded the redraws from above: the statistic can add exclusions relative to 4.21. A sample whose rate sits at the top of the calibration grid has an unresolved rate (it may be larger): its dyads receive the working score with `certifiable` false and stay compatible for the count decision; the exclusion summary reports how many samples are flagged (`unresolved_rate_samples`).

Measured on truth (lab422/r4a, 4.21 release-matrix and G1i saved roots): with independent sites and per-sample redraws, heterogeneous, clustered and misspecified arms included, no true parent was excluded among 2,556 true parent-offspring pairs, with margins unchanged from 4.21. Two departures from the cell model are not absorbed by a per-sample rate. Shared reads make runs of sites overconfident (on 30-kb simulated libraries, a few molecules from one homologue read a heterozygote as homozygous at consecutive sites), and the score then excluded 7 true parents; all were in libraries flagged for read sharing below, which never certify. A long hemizygous tract (a deletion carried by parent and child) excluded a synthetic true parent from about 435 opposite-homozygote sites over 0.7 Mb at 10.5x (4.21: about 1,260 over 1.45 Mb).

The exclusion identity binds the per-sample rates, the unresolved-rate flags and the read-sharing flags.

The e-value is only as valid as the plug-in rates. At confident homozygous sites an understated rate inflates the per-site null expectation by about the ratio of the true to the modelled rate; in a simulation with rates understated tenfold at depth 25, a true parent reached 9.0 nats against a 10.4-nat threshold. The rates are calibrated on the painting's sites and applied to every evidence site; that transfer is an assumption, not a measurement.

The scalar exclusion e-value is invalid under read sharing because successive site bets can depend on the same
molecules. In the 4.21 LR replay the sound physical-split bound made every bet after the first grid row at most one
on a contig with a sharing sample, giving no practical exclusion power there. For longbc F1_14 against F3_21,
the guarded genome upper endpoint was −8.283 nats at a threshold of 9.680 nats, versus a scalar lower endpoint
of 12.396 nats. The guarded scan used 3,341 CPU seconds on TroMau, or 11.37% of pedigree CPU.

A sample is flagged when its library has survival status `fitted` in the painting calibration's `read_libraries`
record on any analysed contig. A dyad with either sample flagged receives the full scalar working score with
`certifiable` false. It remains compatible for the count decision regardless of its score. Two C2 samples, in
libraries without detected sharing, use the existing scalar e-value and certificate. Setting `[painting] read_classes = 1` explicitly bypasses read-sharing
detection. Scalar exclusion certification then depends on the configured conditional site-independence
assumption, including when sequenced fragments span several sites. The exclusion record identity
binds the sample flags.

Implementation: [stages/pedigree/exclusion.py](../../haplotype_reconstruction/stages/pedigree/exclusion.py), [science/pedigree/exclusion.py](../../haplotype_reconstruction/science/pedigree/exclusion.py), [science/pedigree/exclusion_bound.py](../../haplotype_reconstruction/science/pedigree/exclusion_bound.py).

Parameters by name: `alpha`, `bet_sizes`, `window_widths_bp`; the per-sample outlier rates are the painting calibration's (`robustness_grid`). Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(D L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ps21"></a>
## ps21 Completed-window exclusion certificate

Visit sites in order, saving starting-window prefixes before evaluating completed windows in stable stop order. For a certifiable dyad, subtract the declared numerical guards and stop early only when the completed-window lower bound plus the contig's log weight proves the genome threshold (log w_k + log E_k >= threshold). Finish and reduce bets in the reference order when early certification is unavailable or the bound falls short. Store exact values as point intervals and certified values with an unbounded upper endpoint. Publish exclusion only when count-release target dyads exist.

Implementation: [science/pedigree/exclusion_bound.py](../../haplotype_reconstruction/science/pedigree/exclusion_bound.py), [science/pedigree/run.py](../../haplotype_reconstruction/science/pedigree/run.py).

Parameters by name: `alpha`, `bet_sizes`, `window_widths_bp`, `exclusion_dyad_batch`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: At most the full bet/window dyad scan; certification can stop a dyad after a completed window proves the threshold.

<a id="ps22"></a>
## ps22 Scoring orchestration, records, identities and resume

Publish prepared, selective-screen, pair-bound, piece-score and exclusion records under their distinct checked identities. A score identity binds the scoring parameters, the material partition rule and the contig's placement on the genome axis (offset, last unit and cut list), never the regime or release thresholds; an exclusion identity binds the contig's log weight. Reuse the selective screen as a whole, other products at their declared contig granularity, and rebuild the deterministic candidate panel from records. Keep state counters distinct from timing diagnostics. Worker count, completion order, resident caching and memory pacing cannot alter scientific values or admission.

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
## pd05 Exposure gate and units

Require every named edge and M2 pair to meet inclusive exposed-bin, exposure-fraction and unit thresholds, with no tie band. Each exposed material unit counts one, once even when bootstrap weight repeats it. An untestable row retains M0 as its state score and has no finite identity score. Validate edge membership and required exposure fields.

Implementation: [science/pedigree/selection.py](../../haplotype_reconstruction/science/pedigree/selection.py).

Parameters by name: `minimum_edge_exposed_bins`, `minimum_pair_exposed_bins`, `minimum_exposed_fraction`, `minimum_exposed_units`, `preparation`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(H (E + T)). These terms describe the unit's work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="finite-continuous-direction"></a>
<a id="pd06"></a>
## pd06 Joint-transmission orientation evidence and the direction correction

For an ordered dyad (child c, parent p), score the joint likelihood of their four scheduled homologues.
One homologue of c is a meiotic copy of p's two scheduled homologues. The hidden state is (h, s): which child
homologue is copied, and which parental homologue the meiotic selector carries. At the start of each component
these four states have independent uniform probability 1/4. Direction-common founder starting factors are omitted.
At a bin boundary, the selector stays with probability 1 − theta and switches with probability theta, using the
prepared meiotic selector probabilities (numerically bounded by `transmission_switch_floor` and 1/2).

At an informative bin the emission is 1 − m if the copied child's and selected parent's local painted classes
agree, and m otherwise, with the bin's allowance m = min(1/2, max(0.01, v_c + v_p)): v the painting's per-bin
label-error statistic of each sample's matched homologue (the reference painting's posterior allocation of expected
lost copies; 4.21 h, A9, read from the painting's untempered chain). The floor 0.01 is `DirectionParameters.mismatch_probability`, 4.20's constant. These
allowances are data-dependent working scores: penalties stay finite and one bin's contrast is bounded, but no union
bound or validity under any dependence is claimed. If either sample
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

For each material unit compare the forward and reverse joint log likelihoods, d = log L(c ← p) − log L(p ← c).
A unit's contrast carries direction only through the transmitting meiosis: where the transmitted homologue shows a
crossover the reverse direction pays mismatches and d is large; where it shows none, forward and reverse explain
the same mosaics and d is near zero, unless painting noise on the shared homologue makes it large either way. A
decisive contrast is therefore the transmitting meiosis with some probability rho and a fair coin otherwise, and
rho depends on the paintings. With epsilon the unit's painting-failure probability and pi = (1 − epsilon) rho, the
unit's robust orientation log odds are

`v = log[pi sigmoid(d) + (1 − pi)/2] − log[pi sigmoid(−d) + (1 − pi)/2]`,

capped at ln((1 + pi)/(1 − pi)). epsilon follows the relationship model's two-state failure process on the
working map ([PD28](#pd28)): failure regions start at lambda = `failure_region_rate` per Morgan and end at
mu = 1/`failure_region_length` per Morgan; from its stationary law the probability that a region touches a unit of L
working Morgans is epsilon(L) = 1 − mu/(lambda + mu) exp(−lambda L) (a 1-Morgan unit: 0.0997). With rho = 1 a
1-Morgan unit is capped at 2.95 nats; with rho = 0.76, a typical value on low-coverage simulated paintings, at 1.68.
Scope of epsilon: it protects against the short painting failures of the relationship model only. The optional
window cap bounds a chromosome-wide failure's votes without counting chromosomes: with `window_morgans` W > 0 the
units are grouped into consecutive windows of W working Morgans along the unit axis and each window's summed log
odds is capped at `window_cap` (ln 39) before the windows are summed; it is off by default (W = 0).

A linked pair's units are either all informative or all fair coins. They are coins when the pair is a symmetric
relative that the configuration evidence links in place of a parent (full sibs, half sibs and cousins vote a fair
coin at every magnitude of d), or a parent-offspring pair whose crossovers are invisible in the paintings. Let eta be
the run's share of linked pairs whose units are coins, e = `pair_failure_probability` the pair-level failure, Pf the
product over units of the probability of each unit's sign under the forward orientation, Pr its mirror image and
C = 2^−n. The pair law is

`P(D | forward) = (1 − e − eta) Pf + (e/2)(Pf + Pr) + eta C`,

and its mirror image for the reverse. A fit with unit multiplicities w needs two unit-additive sums, L = sum w v and
G = sum w log cosh(v/2): the two orientations' log likelihoods against a coin are A = L/2 − G and B = −L/2 − G, and
the fit's effective orientation log odds are

`L_eff = log[(1 − e − eta) e^A + k] − log[(1 − e − eta) e^B + k]`, `k = eta (e (e^A + e^B)/2 + 1 − e)`.

The direction correction of the family terms, ln 2 + ln((1 − e) sigmoid(L_eff) + e/2), is then exactly
log[P(D | forward)/P_ref(D)] with P_ref the mean of the two orientations: direction evidence splits the support of a
link between its two orientations and never changes the support of the link. Coin data drive A and B down linearly
in the number of units while L only wanders as its square root, so a symmetric relative's random walk does not
accumulate (L_eff tends to zero); an informative pair keeps L_eff near L. Because L and G are sums over units, the
leave-one-unit-out shortcut and the bootstrap multiplicities stay exact.

**The run estimates rho and eta from its own linked pairs.** No constant states how reliable a run's paintings
are. One additional full-data fit leaves every pair's direction evidence out; the pairs named in its samples' local
configurations are the reference pairs: the pairs the configuration evidence links as parent and offspring. The same
fit gives each reference pair a structural orientation S0 (its orientation log odds without direction evidence),
used as the prior w = (1 − e) sigmoid(S0) + e/2 of the pair's orientation. The marginal likelihood of the reference
pairs' unit signs given their magnitudes is

`sum_i log[(1 − eta)(w_i Pf_i(rho) + (1 − w_i) Pr_i(rho)) + eta C_i]`.

With uniform priors on rho and eta, the estimates are the posterior means, summed on fixed grids of midpoints
(`direction_reliability_grid` by `direction_coin_share_grid`; no optimiser, no random draw). For a pair of known
orientation the count of right votes is binomial with the informative probability, and with probability 1/2 for a
coin pair: a two-component binomial mixture, identified once pairs have three or more compared units (every assessed
run has at least five). The likelihood is flat where rho tends to 0 or eta to 1; both state that the run has no
direction evidence. With few reference pairs nothing is replaced by a default: without any, the posterior is the
prior (rho = eta = 1/2: a 1-Morgan cap of 0.97 nats and half the pairs presumed coins), and n clearly informative
pairs leave eta near 1/(n + 2). A run that cannot show that its direction data are informative gets little
direction evidence. The estimate reads no sample name, cohort, relatedness family or released edge, and the
reference pairs are not treated as true pairs: the mixture decides which of them are informative. rho and eta are
run constants, estimated once on the full data; every resampled fit conditions on them, as on the preparations.
They are recorded with the junction evidence (`reliability`, `coin_share`, `reference_pairs`,
`reference_units`).

Require positive bins and callability. Jointly impossible or exactly equal directions contribute zero. Sum in
unit order with resampling multiplicities, then apply the pair-failure direction correction to finite configuration
scores, which the birth-order layer ([PD08](#pd08)) then reads. Resampling and release tiers retain their existing
definitions; the relationship guard is defined in [PD23](#pd23). The junction-density `order_probability` in [PD07](#pd07) still supplies graph search
order and reports; it is a separate quantity. This intermediate direction evidence differs from the released
orientation log odds ([PD08](#pd08)).

**Identifiability limit.** A parent-offspring pair sharing one unrecombined homologue on every chromosome is
near-neutral under this evidence. Alternative mismatch paths can leave residual direction odds of a fraction of a
nat rather than exactly zero. The shared homologue itself does not identify direction, and the trio likelihood
decides these cases. A child homologue that recombines the parent's two can supply direction. Painted junction
totals do not identify biological ages in an inferred founder frame.

**Limits of the direction evidence.**

- *Ancestral relatives vote like parents.* Grandparents, double grandparents and uncles or aunts point
  ancestor-to-descendant in 64 to 75 % of their units on known truth. Linked in place of a parent they count as
  informative pairs, lower the run's rho slightly and stay oriented. No direction statistic separates them from
  parents.
- *Few units cannot tell a sibling's run from a parent's votes.* Five equal signs have probability 1/16 under a
  coin. In small families at 5 to 8 Morgans a released single-parent edge can be a full sib whose few units happened
  to agree; such edges should be read as "parent or close relative". The coin component removes the accumulation
  over many units, not this.
- *Pedigree loops.* For the child of a parent-offspring mating the child's other homologue is also identical by
  descent with the parent, the reverse transmission explains the paintings as well, and the unit votes can oppose
  the true orientation. The pair-level failure keeps the direction correction within ln((2 − e)/e) = 4.60 nats,
  and the configuration evidence of such children (two sequenced parents) decides.
- *The released orientation follows from the birth-order posterior.* The released orientation log odds
  ([PD08](#pd08)) are log P̂(p born before c) − log P̂(c born before p) over the full fit's kept sweep-end orders,
  counts smoothed by half, so bounded at ±8.29 nats with 2000 kept sweeps. A named parent precedes its child in every
  order consistent with the row, so P(p before c) ≥ P(X_c names p): an edge whose full-data posterior is at least
  0.98 passes the orientation cut (`orientation_reversal_probability` 0.02, 3.89 nats) by construction, and for such
  edges the cut is no independent check; the bootstrap and leave-one-unit-out support cuts remain the independent
  checks. For moderate posteriors the cut still binds. The direction model's structural prior (above) reads the same
  statistic from the neutral fit. Where the configuration evidence orients a pair wrongly and the direction evidence
  is weak, the edge can still be released reversed: honestly weak direction evidence cannot veto it. On the 16
  known-truth replay roots of the 4.22 build (N = 60–120, standard and low material) Tier B at the decision had no
  direction error apart from one known edge of the zero-stress design; the 4.21 statistic's wrong-sign rate on
  calibration layouts (0.13 % of linked true pairs) belongs to a different statistic and was not re-measured for this
  one.
- *Few linked pairs.* With few reference pairs rho and eta stay near their priors and the run gets little
  direction evidence. The estimate conditions on the full data; resampled fits do not propagate its uncertainty.

Validation uses constructed schedules, independent Python path sums, homologue-order symmetries, saved likelihood
and decision replays, and known-truth simulations: the run's estimate against the same fit on true pairs with
known orientation (median absolute difference in rho at most 0.001 in every design and material band of the
calibration layouts except small families at low material, 0.014), and the stated direction support against truth on held-out simulations. Inspect lost, gained
and reversed true edges separately: net edge counts can hide losses. Direction log odds determine pedigree calls,
release tiers and edges, and can affect subsequent phase and recombination outputs.

Implementation: [science/pedigree/transmission.py](../../haplotype_reconstruction/science/pedigree/transmission.py),
[science/pedigree/orientation.py](../../haplotype_reconstruction/science/pedigree/orientation.py).

Parameters by name: `mismatch_probability`, `transmission_switch_floor`, `failure_region_rate`,
`failure_region_length`, `window_morgans`, `window_cap`, `pair_failure_probability`,
`direction_reliability_grid`, `direction_coin_share_grid`. Values and rationale are in the
[generated parameter reference](../reference/parameters.md).

Cost and scope: O(D B) for the likelihood and O(H U) for the unit contrasts and their log cosh (one more H by U
array in memory, not stored); the run's estimate costs one full fit without resampling and O(H P G) arithmetic
for P reference pairs and G grid points. The likelihood has four states, independent of founder count once the
schedules are built.

<a id="pd07"></a>
## pd07 Junction-density order probability (graph visit order, role probability)

The contrast, variance and comparison-count condition apply to the material-unit rows in every regime. LOBO
removes the omitted unit's four additive moment contributions before normalization.

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Compute a weighted junction-density contrast over material units with common callability, including missingness variance. Insufficient total weight leaves a neutral order probability; otherwise use the weighted sum, not its mean, in the normal approximation. This retained heuristic supplies the role probability and the reporting order of the selected rows; it decides no row.

Implementation: [science/pedigree/orientation.py](../../haplotype_reconstruction/science/pedigree/orientation.py).

Parameters by name: `order_minimum_unit_weight`, `order_variance_offset`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(C N²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd08"></a>
## pd08 Birth-order layer: the latent birth order inside every fit

Every sample has an unknown birth order π, all N! total orders equally likely a priori, and a parent is born before its child. Given π, a configuration row r of child c is admissible when every parent it names precedes c; rows that name no parent (M0) and exposure-untestable rows are always admissible. With w_r = exp(score_r + direction_r − log m_r), the configuration's genetic score ([pd04](#pd04), [pd05](#pd05)), the orientation term ([pd06](#pd06)) and m_r the full multiplicity of r's state (states weigh equally here; the configured state priors act in [pd12](#pd12)), the joint posterior is P(π, X) ∝ Π_c w_{X_c} 1[X_c admissible under π]. Over configurations P(π) ∝ Π_c Z_c(π), Z_c(π) the sum of w_r over c's admissible rows; over orders the prior weight of a pedigree is proportional to its number of consistent birth orders (the birth-order prior). Every pedigree consistent with an order is acyclic, so no separate acyclicity rule enters. The prior favours shallow pedigrees: for a child whose named parents are otherwise unconstrained a one-parent row carries 1/2 and a two-parent row 1/3 of its M0 row's prior weight, and a full sib named as a parent costs log 2 against the true pair; the cost of a two-parent row grows with the generations of named parents above it, about log(2L + 1) for L layered parental pairs (log 3 among founders, log 5 for an F2 child, log 7 for an F3 child; log(L + 1) along a single-parent chain). True pairs usually lead the M0 row by tens of nats, so these factors weigh deep against shallow explanations, where the reversal errors of a likelihood alone arise, and rarely decide M0 against a parent row; where they do (a weak late-generation child at low depth or material) the call is conservative, M0 instead of a parent, and such a child is rarely releasable. The layer reads no cohort or sex label; every sample is a candidate parent of every other. It replaces 4.21's reciprocal cavity messages, short-ancestry path factor and greedy acyclic repair.

Each fit (the full data, the direction model's neutral fit, every bootstrap and leave-one-unit-out fit; the prior-sensitivity selections reuse the full fit) samples π exactly by Gibbs sampling: each sweep visits the samples in a fresh random order and reinserts each at an index drawn from its exact conditional, proportional to Π_c Z_c with the sample at that index. After the burn-in each sweep adds the Rao-Blackwellised P(X_c = r | π) = w_r 1[r admissible] / Z_c(π); their mean P̂(X_c = r) gives the configuration scores log P̂ + log m_r, so the marginal state evidence of [pd12](#pd12) is the sampled state posterior. The kept sweep-end orders give every fit's consensus birth order (each sample's mean number of predecessors, its mean position), which the joint read ([pd13](#pd13)) uses, and in the full fit only each pair's order counts, from which the release orientation log odds log P̂(p before c) − log P̂(c before p), counts smoothed by half ([pd06](#pd06)).

Each move is incremental: a sample's own rows are binned by their later-born named parent, and for each child naming it only that child's rows naming it are visited; Z_c without them is the maintained Z_c minus their weight unless that keeps less than `cancellation_guard` of Z_c, when it is recomputed exactly. The N insertion weights come from one linear scan of ratios each at most 1, with no exponential or logarithm; every Z_c is recomputed exactly at each sweep end. Weights stay linear relative to the child's best always-admissible row; a row more than `max_weight_span` above it raises the reference. The full fit's raised references and guarded recomputes are recorded in the product's `birth_order` record. Above about 1,345 nats of margin P̂(M0) underflows to 0 and M0's log evidence reports as −inf; no decision changes.

Each fit draws from its own PCG64 stream (`pedigree.order.chain`), keyed by SHA-256 of its unit multiplicities and its left-out unit under the run's `bootstrap_seed`, in a fixed order (the initial order, one visiting permutation per sweep, one uniform per move): results depend on neither thread count, process count nor task order. The permutations are drawn over the samples in sorted-identifier order, as the relationship reference stream is, so reordering the input's columns does not change the realisation; renaming samples does, so reversal and renaming checks are judged within the seed spread. Monte Carlo error on P̂ is about √(P̂(1 − P̂)/S_eff), about 1 % of P̂ for a row near one half; at 400 kept sweeps the state-evidence margin of a near tie varies by about 0.12 nats between streams, so a child whose resample support sits within one draw of a release threshold can be released under one stream and not another; a run's stream is fixed, so its decisions are reproducible. Exactly tied configurations (equal likelihood and prior weight) are split by the chain's realisation, not routed by the tie protocol ([pd11](#pd11)). Each move equals a brute-force replay of the exact conditional, and long-run estimates equal exact enumeration over all N! orders for N ≤ 7 within Monte Carlo error.

Implementation: [science/pedigree/order.py](../../haplotype_reconstruction/science/pedigree/order.py), [science/pedigree/order_parameters.py](../../haplotype_reconstruction/science/pedigree/order_parameters.py).

Parameters by name: `burn_in_sweeps`, `resample_sweeps`, `full_sweeps`, `cancellation_guard`, `max_weight_span`, `pair_failure_probability`, `bootstrap_seed`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: per sweep O(A + Σ_s A_s + N) in resampled fits and O(A + Σ_s A_s + N²) in the full fit (A rows, A_s rows naming sample s, N² the full fit's pair order counts), about 34 ns per row-sweep: 0.14–0.7 s per resampled fit at N = 80–116 (0.1–2.6 % of 4.21.0's pedigree-stage CPU) and about 3 s at N = 320. One chain is sequential, so a fit runs on one thread; fits are independent and run in parallel across the resampling workers.

<a id="pd11"></a>
## pd11 Tie-aware winners, tie tolerance and the full-fit tie census

Scan rows for the first strict finite maximum and runner-up. Abstain when their margin is within the magnitude-aware winner tolerance, including exact ties of the values compared; never use row index to resolve a scientific winner. Rank-probability groups instead anchor at their first value. The explicit recorder captures only the full-fit/state/informative census; diagnostics never enter resampled selection.

The fits' configuration scores are the birth-order layer's sampled posteriors ([pd08](#pd08)), whose Monte Carlo error is far above the winner tolerance. Configurations tied exactly in the evidence (equal likelihood and prior weight) therefore reach the winner scan already split by the chain's realisation, and the scan abstains only where the sampled values tie. For such ties the complete view ([pd19](#pd19)), the unresolved and tier reason codes and the full-fit tie census depend on the Monte Carlo realisation. Released calls do not: a release needs leave-one-unit-out and bootstrap support, each fit on its own stream, so an exact tie splits the support and stays unresolved, and the count release ([pd22](#pd22)) can only withhold a count on a complete row that disagrees with the supported parents.


Implementation: [science/pedigree/ties.py](../../haplotype_reconstruction/science/pedigree/ties.py).

Parameters by name: `contrast_ulp_factor`, `noise_floor`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd12"></a>
## pd12 Marginal state, then identity

Marginalise configuration scores by state using full eligible multiplicities and configured state priors. First require a unique finite parent-count state, then a unique finite identity within that state. Preserve separate unresolved-state and unresolved-identity reasons. These local choices go to the joint read ([pd13](#pd13)).

Implementation: [science/pedigree/selection.py](../../haplotype_reconstruction/science/pedigree/selection.py).

Parameters by name: `state_priors`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(A). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd13"></a>
## pd13 Joint read

The local choices of [pd12](#pd12) read the birth-order posterior child by child; where they form a DAG they are the fit's pedigree. Marginal choices of different children can still form a cycle (two siblings each naming the other with posterior near one half). The children on a cycle (strongly connected components of the chosen parent → child edges) then take their marginal state and identity among the configurations that the fit's consensus birth order allows ([pd08](#pd08): every named parent born before the child; equal ranks in the run's index convention), and the rule repeats until no cycle remains. Every cycle holds a child not yet restricted (the restricted children's parents precede them in one total order), so each round restricts at least one more child and the result is acyclic. No greedy search or tie-conflict peeling remains. The restricted children are reported (`order_restricted`), and resampled fits count how often the restriction was needed.

Implementation: [science/pedigree/selection.py](../../haplotype_reconstruction/science/pedigree/selection.py), [science/pedigree/graph.py](../../haplotype_reconstruction/science/pedigree/graph.py).

Parameters by name: none (the greedy repair's pass budget is removed).

Cost and scope: O(N + edges) per round, at most N rounds.

<a id="pd14"></a>
## pd14 M0 prior sensitivity

Reselect the full-data birth-order posterior under each declared M0 sensitivity prior, reusing the nominal prior result when equal. Do not resample the birth order. Require agreement for an M0 state and its exact graph configuration at release; a sensitivity veto has explicit status precedence. Resampled fits do not repeat this prior analysis.

Implementation: [science/pedigree/decision.py](../../haplotype_reconstruction/science/pedigree/decision.py).

Parameters by name: `prior_sensitivity`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(A). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd15"></a>
## pd15 Informative units

For each material unit, apply the abstaining winner rule to the child's own parent-count state evidence there. A unit is informative when any state has a unique winner; it need not agree with the genome-wide state or configuration. Count the informative units (each counts one) and report their working Morgans; apply the later release threshold to the count without a tie band. The unit winners are counted in their own tie scope (`pedigree.informative_unit`).

Implementation: [science/pedigree/decision.py](../../haplotype_reconstruction/science/pedigree/decision.py).

Parameters by name: `minimum_informative_units`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(H A). These terms describe the unit's work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd16"></a>
## pd16 Unit bootstrap and LOBO resampling

Draw all unit bootstrap multiplicities first from the named fixed stream, then coalesce identical vectors in lexicographic order. Schedule the leave-one-unit-out (LOBO) fits first, each read from the full totals minus the left-out unit, and bootstrap fits by the declared stable sparsity order. Accumulate integer selection counts with each vector's multiplicity; a failure stops the stage. Scheduling cannot change support fractions. The stage groups consecutive planned fits into batches by estimated alternative-row scans, with a work ceiling and a target of two waves per planned worker when the fit count allows. Each process attaches the read-only inputs once per batch and evaluates its fits serially. The gather restores the complete original plan before reducing integer counts; batching neither redraws multiplicities nor changes denominators. The decision's full-data fit reads the same fit inputs (after the direction model's neutral fit) and nothing of the resampled fits, nor they of it, so it runs as one more single-core task beside the batches and the gather joins it to the counts; each fit's birth-order chain draws from its own keyed stream, so this placement cannot change any result.

Implementation: [stages/pedigree/execution.py](../../haplotype_reconstruction/stages/pedigree/execution.py), [science/pedigree/resampling.py](../../haplotype_reconstruction/science/pedigree/resampling.py).

Parameters by name: `bootstrap_replicates`, `bootstrap_seed`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(H A + H N² (PD07) + (H + R)(birth-order fit, [pd08](#pd08)) + graph). These terms describe the unit's work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd17"></a>
## pd17 Tier reach and the material statement

The reach record states U, the resampled units and the cut values of the [material regimes](#material-units): whether Tier B is assessed (U >= `assessable_morgans`), with which thresholds, and whether Tier A is (U >= `tier_a_morgans` in the standard band). The precheck refuses fewer than three samples only. Before a genome-wide run the command line warns when the configured working map of the analysed contigs (their configured lengths on the run map) lies below a cut; the informative material is measured after painting and is at most that. Reach is a material statement, not a released pedigree call.

Implementation: [science/pedigree/regimes.py](../../haplotype_reconstruction/science/pedigree/regimes.py), [science/pedigree/run.py](../../haplotype_reconstruction/science/pedigree/run.py).

Parameters by name: `assessable_morgans`, `standard_morgans`, `tier_a_morgans`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B) over the prepared layouts. These terms describe the unit's work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd18"></a>
## pd18 Tier A/B state and exact-configuration release

Test state support and informative units first, including M0 prior agreement. Require at least one identified named parent for a non-M0 state. Test parent stability and orientation separately; exact configurations additionally require local/graph agreement, configuration support and prior agreement. All fraction thresholds are inclusive. Preserve status precedence for prior veto, no stable parent, insufficient state support, graph conflict, orientation and identity ambiguity.

Implementation: [science/pedigree/release.py](../../haplotype_reconstruction/science/pedigree/release.py).

Parameters by name: `tier_b`, `tier_a`, `minimum_informative_units`, `orientation_reversal_probability`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="pd19"></a>
## pd19 Complete (full-data acyclic) view

Publish the complete acyclic best-guess view from graph rows, retaining the order-restriction ([pd13](#pd13)), displacement and unresolved-state reasons. A displaced row is not tier eligible merely because it is acyclic. The complete view is not a release and remains unchanged by the sibling guard.

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

Score every candidate of unresolved/count-only Tier-B targets. Exclude a candidate only when its score is certifiable
and reaches the threshold. Reject threshold-straddling intervals on certifiable dyads. Remove compatible reverse directions already supported by nonwithheld outgoing edges. Release a count only with complete evidence, sufficient units, no graph conflict, at most one survivor, agreement among survivors/supported/complete parents and no guard-withheld dependency. Keep resampling fractions unchanged and label the distinct exclusion-based release evidence.

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
a nonzero pair margin is tested even without a family. The support bands (`support_bounds` 0.5 and 0.75) read the
relatedness members' Morgan-weighted support ([RL17](relatedness.md#rl17)); their priors (`sibship_prior`
0.67, 0.96, 0.99) price 0.71, 3.18 and 4.60 nats. They are the full-sib shares of the guard's own witness occurrences
(each priced once per vote) pooled over three known-truth simulation sets (4,159 occurrences in 73 runs; run-bootstrap
intervals 0.54-0.79, 0.91-0.98 and 0.97-1.00, the re-check band); no guard, edge, state or count decision changed under
any held-out table. The shares depend on the design mix and on material: runs below 9.5 Morgans of family-site
material measure 0.38, 0.74 and 0.945, so low-material runs are priced above their measured shares.
A single witness cannot withhold while the pair margin is above
`−(withholding_score − top price)`, where the top price is log(0.99/0.01) = 4.60, i.e. −2.31 nats. Two top-band
witnesses can still withhold. The sibling-guard table reports final witnesses, set-aside/against/supporting counts,
pair/family/total score, final evidence call, irreversible call, withdrawal iteration and pair/family basis.
A witness is against below −1 nat and supporting above +1 nat. Set-aside counts include a contradicted family
check or both hypothesis margins at or below minus its price. Withholding basis is `pair` when the pair margin
alone crosses the negative threshold, otherwise
`family`; a retained edge has `none`.

Recompute all original edges against the retained edge set until no more withdrawals, in at most E+1 passes.
The retained edges then receive the family transmission vote of [PD29](#pd29) once.
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

Values of a few per Morgan reflect the crossovers of the cross's generations: in simulations of supported designs the cohort median is between 0 and 2.6 per Morgan and single samples reach about 7. Values far above that, such as 25–46 in a real cohort with many unrelated founders, indicate fragmented paintings; parent calls based on those paintings are not reliable. These ranges describe interpretation, not decision thresholds. The rate depends on the configured map and scales inversely with its length: doubling a configured recombination rate doubles the denominator and halves this number. An erroneous configured rate therefore rescales the diagnostic.

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

<a id="pd29"></a>
## pd29 Family transmission test of retained parent edges

After the sibling guard's fixed point ([PD23](#pd23)), every Tier B edge p -> c the guard retains receives one more
vote. Its family is c plus witnesses: the guard's tested members of c's relatedness family (members linked by a
retained edge to c or another member removed), minus members failing the guard's family check (the median trio
sib-feasibility margin of (c, w, w') over w's ring partners at or below w's price), minus members with a retained
parent other than p or a retained parent of c. The guard's other set-aside condition (both of its hypothesis margins
at or below the price) is not applied: it depends on p, the hypothesis under test, and it removes the witnesses whose
failure exposes a grandparent. At most `maximum_witnesses` witnesses are kept, by relatedness member support
(descending), then observed-site counts in configured contig order (descending, the [PD24](#pd24) convention), then
the witness's own family-check margins; no sample name enters (complete ties keep the guard's member order). An edge with fewer than `minimum_witnesses` witnesses (a family of fewer than three children) is `untested`
and its statistic is not published: with one witness the evidence is near-pairwise and a single non-sib witness
dominates it.

Three joint models of the family on the PS19 schedules, each with two latent (unsequenced) diploids whose four
homologues are founder mosaics under PD06's law (start 1/T, stay 1 - theta, change to a named other trajectory
theta / (T - 1)):

- PAR: p is a parent: p is a direct observation of latent 0 and every family child has a paternal meiotic selector
  over latent 0's homologues and a maternal selector over latent 1's (the other parent), each switching with the
  prepared selector probability theta (floored by `transmission_switch_floor`);
- REL: p is one more full sib: p and the children are children of the two latent diploids;
- UNR: p is not in the family: the children are children of the two latent diploids and p is its own best mosaic
  (grandparents, double grandparents, half sibs, avuncular and other relatives).

A member's unordered painted pair {x, y} (label 0 = x, label 1 = y) given true classes (u, v) emits the probability
of its labels' error pattern: g(x <- u, y <- v) + g(x <- v, y <- u) for x != y, the first term alone for x = y, with
g = p00 when both labels equal their true classes, p10 when only label 0 differs, p01 when only label 1 differs and
p11 when both differ. The statistic keeps no class-specific mass for a wrong label, so a wrong label's whole mass goes
to whichever class a hypothesis needs: the vote is never more decisive about a wrong label than this law, whatever the
number of local classes (spreading the mass over the K_b - 1 other classes would make each wrong label cost
ln(K_b - 1) more and erode true-parent votes at large K_b). The joint law comes from the painting's label-error
statistic of the member (h, from the untempered chain, on its scheduled homologues): v_t the probability that label t
is wrong and u that the pair is wrong. Since the pair is right exactly when both released classes are carried,
u = P(label 0 wrong or label 1 wrong) and p11 = v_0 + v_1 - u, p10 = v_0 - p11, p01 = v_1 - p11, p00 = 1 - u. The floor
stays because the posterior is overconfident: each marginal lies in [0.01, 1/2] (`DirectionParameters.mismatch_probability`,
PD06's convention), p11 is clipped to what the marginals allow, and every error pattern is at least its independence
value at the floor (p10, p01 >= 0.01 (1 - 0.01), p11 >= 0.01^2), p00 being the remainder. No pattern is zero, so every
emission is positive: a hypothesis needing a wrong label is penalised, never excluded. Without label errors the law is
4.21's, (1 - 0.01)^2, 0.01 (1 - 0.01), 0.01 (1 - 0.01), 0.01^2. The per-bin posterior marginals enter every bin as
emissions; their dependence along the chain is not modelled. Bins without both labels emit 1. A witness may instead be a half child of latent 0 or of latent
1 (one homologue through a meiotic selector over that diploid's homologues, the other a free founder mosaic under
the same law, summed), a direct observation of a latent diploid (a sequenced parent of the family; under PAR only
latent 1) or an outsider explained by its own mosaic, each at log prior ln(`role_prior`), a child at
ln(1 - 5 `role_prior`); c is always a child. Roles are refitted in each segment. The role prior alone thus charges at
most ln((1 - 5 rho) / rho) = 4.55 nats per segment for a witness that is not a full sib; that bounds the prior only:
the decision rests on l_PAR - l_ALT, to which several contaminating witnesses contribute together.

**Half children, and why roles stay per segment.** Without the half-child role, a half sib of c through the other
parent is a full child under UNR in every segment, its foreign homologue absorbed by the spare homologue of UNR's free
latent parent (a chimeric parent), while PAR, whose latent 0 is p's two homologues, cannot explain it at all: about 30
nats per segment against a correct edge whose family holds no full sib of c. The half-child role gives PAR a
comparably cheap explanation of one such half sib. It reduces the chimera but does not stop it: UNR's spare homologue
is shared by every witness that inherited the same homologue of the other parent's other mate, while each half child
pays its own role prior and its own foreign mosaic. A shared second mate for such witnesses was tried and rejected: on
the 4.19 harness it never rescued a correct edge, withheld two more correct edges outside the assessable material and
let six released relatives (half sibs in large sib-rich families) escape, because PAR could then call all of a
child's full sibs half sibs from one mate for a single prior. Roles fixed for the whole genome were tried and
rejected too: chosen by the summed likelihood, a single segment with a painting failure decides every witness's role
genome-wide; chosen with PD06's per-unit robustness, a wrong role passes as capped unit failures when the segments are
few. Per-segment roles keep the statistic's per-unit robustness.

Each painting component is cut into consecutive segments of `unit_morgans` working Morgans (a final piece shorter
than `tail_morgans` joins the previous one; a unit that receives no bin, behind one bin step longer than a unit,
joins the unit before it); each segment is fitted independently. The profile log likelihood of a model maximises
over the latent paths and the witness roles and sums each child's selectors exactly: MAP-EM (exact selector
posteriors; Viterbi over ordered homologue pairs of one diploid with the others fixed, O(T^2) per bin by row and
column maxima), exact homologue-label flip moves (the gain of swapping one diploid's labels after any boundary, from
the pairwise posteriors of its children and half children), role updates, adopt moves (a witness becomes a latent
diploid initialised at its own mosaic, or a half child of either latent diploid), starts seeded at members' own mosaics, `random_starts` block perturbations per round
drawn from the named `pedigree.family_transmission.starts` stream keyed by a digest of the segment's data, and
cross-initialisation of every model from every other model's best paths for up to `start_rounds` rounds. Every step
is monotone; the optimum is local, not certified. Each segment is searched twice, with the witnesses ordered by a
digest of their painted classes and in the reverse order, and each model keeps the better value; the value then
depends neither on the order in which witnesses are supplied nor on sample names (witnesses with identical data are
interchangeable).

With d_u = l_PAR,u - l_ALT,u, ALT the better of REL and UNR summed over the edge's segments, the statistic is the
sum of PD06's robust per-unit log odds of d_u with the PD28 painting-failure probability of the segment's length
(0.0997 and a cap of 2.95 nats for 1 Morgan; shorter segments have larger caps). The edge is withheld when the
statistic is below -`withholding_score` (the guard's, ln 1000). A withheld edge has the guard's consequences (calls
of the child unresolved in both tiers, count bounds [survivors, 2]); its vote is `call = withheld`,
`withholding_basis = transmission`, `withdrawal_iteration` one more than the guard's passes. The vote runs once:
the guard's cleaning is not recomputed after it, and no fixed point is sought. The sibling-guard table publishes
`transmission_call` (kept, withheld, untested; empty on guard-withheld edges), the witness and segment counts, the
statistic and the alternative.

Scope and limits:

- The test acts only where the child has at least two retained witnesses, which in practice means sib-rich and
  small-family designs; it cannot reach a relative released as the parent of a child without sequenced full sibs
  (two-child families, parent-free half sibs), nor separate relatives whose paintings are identical by state over the
  whole family.
- The gain inside the pipeline is unproven: in the 4.19 validation, a clean-gain set of 24 sib-rich simulations
  released no relative for the test to withhold, and the K = 30 run was not assessable. The evidence is the
  harness's released relatives and true relatives scored as candidate parents.
- Power is lower in families of three children whose witnesses include a half sib, and for double grandparents,
  because PAR can explain a witness that did not receive p's homologue as a half child. In simulated three-child
  families (10 one-Morgan segments, 40 replicates; six founder trajectories painted as four classes), the relatives
  withheld were, against the form without the half-child role: a grandparent with one full-sib and one half-sib
  witness 48 % (68 %), with two half-sib witnesses 60 % (80 %), with two full-sib witnesses 72 % (82 %), a double
  grandparent 32 % (52 %); with distinct classes 82 % (95 %), 88 % (90 %), 98 % (98 %) and 65 % (78 %). Larger
  families keep their power (a grandparent with four full-sib witnesses 100 %).
- A residual chimera remains for correct edges whose families hold no full sib of c. In the simulations above, c plus
  four half sibs from one other mate was withheld in 30-62 % of replicates at 6-20 segments (without the half-child
  role 42-95 %), c with its other parent and two half sibs through it in 0-35 %, two half sibs in 0-10 %. On the 4.19
  harness the configuration is rare: of 46,923 tested correct instances, 123 had no full-sib witness, 10 had at least
  two half sibs through the other parent (2 from one mate) and none was withheld; a correct edge with c's parent and
  one half sib as witnesses, just outside the assessable material (4.96 M), scores -9.5.
- The result depends on the search: the optimum is local, and before the two searches another start stream moved
  harness statistics by several nats near the threshold (-6.4 against -9.5 for the edge above); the two searches
  remove the dependence on witness order and names, not on the starts.
- The half child's foreign homologue is summed while latent homologues are maximised, so a half child is cheaper than
  an equivalent latent path; this is what lets PAR match the chimera for one witness.
- The statistic is not a posterior probability.

Implementation: [science/pedigree/family_transmission.py](../../haplotype_reconstruction/science/pedigree/family_transmission.py),
[stages/pedigree/family_transmission.py](../../haplotype_reconstruction/stages/pedigree/family_transmission.py),
[science/pedigree/run.py](../../haplotype_reconstruction/science/pedigree/run.py).

Parameters by name: `role_prior`, `minimum_witnesses`, `maximum_witnesses`, `maximum_iterations`, `start_rounds`,
`random_starts`, `mismatch_probability`, `transmission_switch_floor`, `failure_region_rate`, `failure_region_length`,
`withholding_score`, `unit_morgans`, `tail_morgans`. Values and rationale are in the
[generated parameter reference](../reference/parameters.md).

Cost and scope: per tested edge and segment of B_s bins with k children and T trajectories, O(R I B_s k T^2) for R
fits of I EM iterations, with O(k) adopt fits each, so the cost is quadratic in the capped witness count, and two
searches per segment; witnesses are capped, so the per-edge cost is bounded independently of family size. Per-contig
work runs as independent tasks after the guard's fixed point and is not checkpointed.

<!-- named mechanism anchors -->


## Named mechanisms

<a id="birth-order-layer"></a>

**Birth-order layer:** see [PD08](#pd08).

<a id="birth-order-prior"></a>

**Birth-order prior:** see [PD08](#pd08).

<a id="joint-read"></a>

**Joint read:** see [PD13](#pd13).

<a id="candidate-panel"></a>

**Candidate panel:** see [PS15](#ps15).

<a id="child-forward-model"></a>

**Child forward model:** see [PS10](#ps10).

<a id="unit-resampling-mechanism"></a>

**Unit resampling:** see [PD16](#pd16).

<a id="configuration-likelihoods-and-the-exposure-gate"></a>

**Configuration likelihoods and the exposure gate:** see [PD04](#pd04).

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

<a id="preparation-markers-and-source-factors"></a>

**Preparation markers and source factors:** see [PS02](#ps02).

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
