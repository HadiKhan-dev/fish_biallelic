# Family phase

Family phase conditions on frozen painting, refit alleles and released parent edges. Two convention solves optimise a founder-mosaic objective; release requires their agreement while preserving uncertainty and phase-set boundaries. N is samples, B bins, L sites, S labels and E parent edges. The modular conditional model does not refit upstream ancestry or pedigree. A converged coordinate ascent need not be a global optimum.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="fp01"></a>
## fp01 Component model inputs: bins, labels and tempered pair emissions (cache or recompute)

Build symmetric pair emissions using the painting record’s estimated robustness_epsilon and its floor and calibrated transition site correlation. Use its packed cache or recompute the same binned emissions and design effect, including empty-information bins. Require painting/pedigree sample alignment. Both solves share these immutable pre-refit emissions; later allele refit does not change this ancestry objective. For samples sharing reads, the confidence-weighted painting matrices supply a [class-message projection](./family_phase.md#read-class-evidence) to the bin emissions.

Implementation: [contracts/phase.py](../../haplotype_reconstruction/contracts/phase.py), [science/painting/emissions.py](../../haplotype_reconstruction/science/painting/emissions.py), [science/phase/component.py](../../haplotype_reconstruction/science/phase/component.py), [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py), [science/phase/painting_inputs.py](../../haplotype_reconstruction/science/phase/painting_inputs.py), [stages/phase/records.py](../../haplotype_reconstruction/stages/phase/records.py).

Parameters by name: `SITE_WINDOW`, `robustness_grid`, `emission_log_floor`, `divisor_grid`, `EMPTY_COMPONENT_BIN_SITES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N S^2 B); O(N S^2 L_c). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp02"></a>
## fp02 Edge rates and initial terms (label switch, Haldane meiosis switch with protected resets, not-a-copy entry/exit)

Derive label-switch rates from recorded painting transitions, meiosis-switch rates from map distances and the shared Haldane fraction, and not-a-copy entry/exit rates from physical distance. Protected intervals force meiosis resets. The shared rate transform supplies the Haldane formula. The not-a-copy entry and exit laws use their own physical-distance rates. Initialise linked slots with the stationary copy/not-copy terms and preserve the reset boundary test.


Implementation: [science/phase/inputs.py](../../haplotype_reconstruction/science/phase/inputs.py), [science/shared/recombination.py](../../haplotype_reconstruction/science/shared/recombination.py), [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py), [science/phase/model.py](../../haplotype_reconstruction/science/phase/model.py).

Parameters by name: `not_copy_rate_per_bp`, `not_copy_mean_bp`, `alpha`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp03"></a>
## fp03 Joint objective J, per-sample terms and local gains

Sum each sample’s emission terms in bin order, then each homologue’s transition terms. For a proposed block or pairing move, compute local gain with an accurately rounded sum of after-minus-before terms for exactly the changed rows; no changed row gives exact zero. This local gain decides acceptance. The full objective is a verification quantity and may round differently near a threshold.

Implementation: [science/phase/kernels.py](../../haplotype_reconstruction/science/phase/kernels.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(R B); O(N B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp04"></a>
## fp04 Exact sample block (sources + children's gauge) and its Viterbi tie rule

Solve the sample’s exact source/gauge block with every other sample fixed. Start gauge at zero. At each predecessor stage keep the current source within the tie threshold; the gauge prefers continuity at inner bins. At the final bin prefer the current sources with zero gauge. The retained argument’s score, not the numerical maximum, continues the recurrence. Scan reversal changes only remaining arbitrary preferences. Propagate descendant labels and reset scratch marks after use.

Implementation: [science/phase/kernels.py](../../haplotype_reconstruction/science/phase/kernels.py).

Parameters by name: `tie_quantum`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(G (n0^2 n1 + n0 n1^2 + n0 n1 G)); O(n0^2 + n1^2); O(G |desc| S). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp05"></a>
## fp05 Pedigree order, descendant sets and children lists

Compute ancestry depths, require acyclicity and visit ancestors before descendants. Within a depth, use the solve’s explicit index convention. Sort child tracks by their encoded track index and descendant/child solve lists by pedigree rank. These orders determine Gauss–Seidel updates and descendant arithmetic; they are not interchangeable scheduling choices.

Implementation: [science/phase/model.py](../../haplotype_reconstruction/science/phase/model.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N x depth); O(sum_n |desc_n|). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp06"></a>
## fp06 Two-pass initialisation

First solve every sample in index order with both labels free and no child terms, using zero-initialised label incumbents in both conventions. Then revisit linked samples in ancestry order with their parent kinds and no child terms, initially copying parental homologue zero. Apply both passes without a gain check. Initialisation is free-label Viterbi on tempered painting emissions, not a copy of painting MAP tracks.

Implementation: [science/phase/model.py](../../haplotype_reconstruction/science/phase/model.py).

Parameters by name: `tie_quantum`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B S^3). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp07"></a>
## fp07 Block coordinate ascent sweeps (acceptance, quantum, cap, settled)

Append the starting objective, then sweep samples in pedigree order. Save affected rows, solve/apply the block and accept only a local gain strictly above the gain quantum; otherwise restore. Later blocks see accepted updates. Append the objective after every sweep. No accepted block means settled; reaching the cap with acceptances means unsettled. The reported component requires both solves’ final sweep calls to settle.

Implementation: [science/phase/model.py](../../haplotype_reconstruction/science/phase/model.py).

Parameters by name: `gain_quantum`, `max_sweeps`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B S^3); O(B (1 + A_n)); O(N B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp08"></a>
## fp08 Parents' pairing joint moves, solve loop and pairing ambiguity

After initial sweeps, enumerate eligible unlinked parents and their double-switch junctions once per pairing pass. Swap the parental suffix, propagate and re-solve children in rank order, then keep only a strict gain above the maximum of pairing odds and gain quantum. Restore rejected moves and record final-state ambiguity. Accepted passes trigger more sweeps; reaching the pass cap adds a keep=False pass so ambiguity is measured without another update. Compute pairing odds from the posterior parameter, never a rounded literal.

Implementation: [science/phase/model.py](../../haplotype_reconstruction/science/phase/model.py).

Parameters by name: `set_posterior`, `gain_quantum`, `max_sweeps`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n_children B S^3); O(B (1 + A_p)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp09"></a>
## fp09 The two convention solves (default reverse=False and mirror reverse=True)

Run default and mirrored convention solves independently. The mirror reverses within-depth visits, nonincumbent tie scans and ascent descendant order, while retaining initial label/copy incumbents, gauge preferences, thresholds and first-pass sample order. Agreement always releases the default solve’s orientation, even when the run’s reporting convention is reversed.

Implementation: [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Two independent ascents with shared immutable emissions; release evaluates both final states.

<a id="fp10"></a>
## fp10 Identity junction candidates and withheld joins (pin test)

Find shared identity-junction candidates by distinct switching samples, counting a sample once even when both tracks switch. Withhold a candidate join unless a linked slot copies the same parental homologue on both sides and the parent’s reads pin both flanks. Stop each flank at a parental label change and use inclusive pin evidence. Children do not pin. Union withheld joins from both solves; later tie gates cannot bridge them.

Implementation: [science/phase/junctions.py](../../haplotype_reconstruction/science/phase/junctions.py), [science/phase/model.py](../../haplotype_reconstruction/science/phase/model.py).

The pinning flank spans junction_pin_reference_bins reference windows, or that count times the painting’s reference-to-fine bin ratio in fine bins, stopping at parental label changes. Recomputed phase emissions use the same per-sample reference-window divisor as painting.

Parameters by name: `junction_min_samples`, `junction_pin_nats`, `junction_pin_reference_bins`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N B); O(candidates x N); O(window S). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp11"></a>
## fp11 Painting tie boundaries

At consecutive painting chunks compare direct and swapped continuation by the lexicographic pair of label changes and allele-visible changes. Treat every case except strictly better direct continuation as a tie boundary. Map it to the next site’s bin and retain only internal boundaries of samples with no linked slot. A sample with one released parent does not receive this free-sample tie rule.

Implementation: [science/phase/painting_inputs.py](../../haplotype_reconstruction/science/phase/painting_inputs.py), [stages/phase/records.py](../../haplotype_reconstruction/stages/phase/records.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N x chunks x K). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp12"></a>
## fp12 Per-sample forced-break assembly and break counters

Union painting ties, withheld joins and ambiguous pairing boundaries from both solves for each sample. Pairing boundaries propagate to the parent and its children. Track tie-only cuts separately so the child-anchored gate can bridge only those. Count overlapping break reasons without double-counting the specified join category. These forced breaks bound phase and meiosis release.

Implementation: [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N x breaks). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp13"></a>
## fp13 Released site axis, site bins and state alleles

Align refit panel positions to the evidence axis and painting bins. Release only refit sites lying inside a painting bin. A named state has an allele only when all its class rows are called and agree; BACKGROUND stays unknown. Other evidence sites keep missing calls. Retain component geometry even for components skipped by inference so later chain columns stay aligned.

Implementation: [science/phase/inputs.py](../../haplotype_reconstruction/science/phase/inputs.py), [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(L + S m). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

### Site-specific wrong-allele reads

Family phase estimates a recurrent wrong-read rate at each QC-kept site, using the
published painting and the original founder panel. The two named tracks and their
called panel alleles supply a projected genotype. Only painted homozygotes with
reads contribute to the fit. Libraries excluded by sample QC or fitted with a
separate contamination correction supply no fitting evidence.

The likelihood for a contributing cell averages its genotype likelihoods against
a working prior. The prior assigns weight `1 − b` to the painted homozygote and
weight `b` to the existing cohort genotype distribution, where
`b = 1 − (1 − ε)(1 − u)`. Here `ε` is the painting's outlier rate, and `u` is
the finite released-pair error in that bin, or zero when unavailable. This
allows uncertainty in the painting to explain recurrent minor reads.

Each alternative changes one homozygote's wrong-read probability from the fitted
library rate `e` to `e + (1 − 2e)h`. The opposite homozygote and the native
heterozygote balance and dispersion retain their original laws. The slab gives
equal weight to both directions and the eight dyadic values of `h` from
`1/1024` to `1/8`. Family phase sums the log likelihood ratios over eligible
libraries and fits the slab prevalence by maximizing the contig's marginal
likelihood. The slab is retained only when its gain exceeds half the log of the
number of kept sites. A direction needs a strict posterior majority and uses
its conditional posterior mean rate.

Family phase uses the resulting site likelihood for carrier allele evidence,
own-read subtraction, flips, and the ordered posterior in both solve orders,
including whole-bin and anchored orientation calculations. Sites without a
selected direction and libraries under a separate contamination correction
retain their native likelihoods. The binned emissions and solves use their
original observation model. The fitted rates are conditional on the painting
and cohort frequencies, which can themselves use the focal reads.

Implementation: [science/phase/site_error.py](../../haplotype_reconstruction/science/phase/site_error.py), [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py).

Parameters by name: `site_error_rates`, `site_error_bic_penalty`, `site_error_direction_posterior`, `SITE_ERROR_BATCH`, `SITE_ERROR_BISECTIONS`, `SITE_ERROR_PREVALENCE_BOUNDARY`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

<a id="fp14"></a>
## fp14 Panel-allele uncertainty (prior, carrier evidence, flips, own terms)

Form the chromosome-wide allele prior from all called refit cells. Accumulate robust carrier evidence from solve-MAP labels, excluding founder-excluded samples, in fixed site chunks and homologue-then-sample order. Convert it to capped allele-flip uncertainty; unknown alleles have no flip correction. Remove the released sample’s own terms by subtraction before its site model. Resident/windowed routes preserve the same float32 evidence inputs and float64 accumulation.

Implementation: [science/phase/allele_evidence.py](../../haplotype_reconstruction/science/phase/allele_evidence.py).

Parameters by name: `FLIP_NEGLIGIBLE`, `FLIP_CEILING`, `PRIOR_BOUNDS`, `EVIDENCE_CHUNK`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N m); O(m); O(S m). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp15"></a>
## fp15 Block posterior (forward-backward), label-pair posterior and chain marginals

Compute the exact conditional block forward/backward posterior with gauge starting at zero and any anchor mask enforced in both directions. Sum gauge and other-slot states in declared row order, applying the small-log-mass cutoff without renormalisation. Release uses topological depth/ascending-index descendant order in both solves, independent of workers. Retain chain probabilities as float64 until release tests, casting only stored support.


Implementation: [science/phase/kernels.py](../../haplotype_reconstruction/science/phase/kernels.py), [science/phase/release_kernels.py](../../haplotype_reconstruction/science/phase/release_kernels.py).

Parameters by name: `LABEL_SWITCH_MINIMUM`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B (S + 2)^3 G). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp16"></a>
## fp16 Site model (ordered allele posteriors per site)

Mix known label-pair mass with per-copy allele uncertainty and the site-deviation law. Preserve exact-zero grouping: unchanged pairs, one-flipped-label sides and individually normalised two-flipped pairs are accumulated in their specified order. Skip unknown pairs or nonpositive normalisers. Validate posterior settings against the model’s attainable certainty ceiling. Sites are independent conditional on the block posterior, but the block remains pedigree conditioned.

Implementation: [science/phase/kernels.py](../../haplotype_reconstruction/science/phase/kernels.py), [science/phase/release_kernels.py](../../haplotype_reconstruction/science/phase/release_kernels.py).

Parameters by name:. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(m S^2); O(m (4 + 4F + F^2)); O(K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp17"></a>
## fp17 Genotype and per-site phase calls

Call genotype from the unanchored genotype posterior, with the lower genotype winning an exact maximum tie and an inclusive call threshold. Use the ordered posterior to phase called heterozygotes at its own inclusive threshold. A strict P10 advantage gives reversed allele order; otherwise retain P01 order even for an unphased heterozygote. Uncalled sites remain missing and frame uncertainty is explicit.

Implementation: [science/phase/release.py](../../haplotype_reconstruction/science/phase/release.py).

Parameters by name: `genotype_posterior`, `phase_posterior`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(m). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp18"></a>
## fp18 Gauge-sample anchor partitions and orientation-certainty set ends

For an unlinked sample without child tracks, partition at hard cuts and choose the first confident heterozygous anchor in each interval. Anchor so that homologue 0 carries the label ranked first, and recompute the conditional posterior. Labels are ranked by how closely their panel alleles at the anchor bin match homologue 0 of the bin's most probable label pair taken in label order: over the bin's sites where that pair's two alleles are known and differ, sites matching its first allele minus sites matching its second, with unknown alleles counting for neither. Equal scores keep the pair's first label first and its second label last among the labels that score as they do, so a label carrying the same alleles never splits a pair label across homologues; other ties go by label index. Where the pair's alleles never differ in the bin, the alleles give no frame and labels are ranked by index. The ranking is a strict total order, so the anchor is still a gauge: unordered posteriors, genotype calls and missingness are unchanged, and the most probable pair keeps its label order. The ranking fixes the allele frame where the anchor's posterior splits between pairs that share a label in swapped sorted positions. Ranking by label index alone would put that label on opposite homologues there, and the mixed frame would end the set at the next tested bin and at every re-anchor after it. Where no posterior mass sits on pairs that the ranking orders against label order, the anchor selects the same mass as ranking by label index. End the set at the first later sufficiently heterozygous bin losing orientation certainty; NaN certainty does not trigger that stop. Include pre-anchor bins in the anchored partition and compute orientation medians over whole bins. The orientation certainty uses the set statistic's own site model, a departure `set_site_deviation` against a uniform genotype, rather than the run's outlier rate: under that rate a fully determined common heterozygote caps below `set_posterior`, and every heterozygous bin ended a set on real data. The parameters refuse a `set_posterior` at or above the ceiling (1 − 5d/6)/(1 − 2d/3). The median runs over the bin's witness sites: the sites confidently heterozygous under that model, except those that the labels alone (the same site model with no departure) make confidently homozygous, each with its certainty under the set model. A site the labels alone make homozygous is heterozygous only through the departure: it has equal probability of either allele order under every frame and is no evidence about the frame, and in a bin with one or two confident sites it would otherwise decide the median. Every other confident site stays, so a label pair whose alleles cannot produce the reads still enters at 50/50 wherever another candidate pair explains part of the heterozygote, which keeps the departure's guard against near-ties at read-incompatible label pairs. A bin without a witness is untested. A partition without an anchor releases no phase. After each anchor, forward lookahead starts at one bin and doubles after a piece passes the continuation test, up to the bounded work per piece. Each bin uses the same posterior recursion.

Implementation: [science/phase/release.py](../../haplotype_reconstruction/science/phase/release.py).

Parameters by name: `phase_posterior`, `genotype_posterior`, `set_posterior`, `set_site_deviation`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(m_partition S^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp19"></a>
## fp19 Children-anchored painting-tie gate (B)

For an unlinked sample with children, inspect finite orientation bins immediately before and at/after each tie-only cut. Bridge only when both sides reach the set-posterior threshold, with the same set statistic as fp18: its site model and its witness sites. Each solve decides separately and agreement keeps any break either retains. Linked samples and gauge-only samples do not use this gate.

Implementation: [science/phase/release.py](../../haplotype_reconstruction/science/phase/release.py).

Parameters by name: `set_posterior`, `set_site_deviation`, `genotype_posterior`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(m S^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp20"></a>
## fp20 Agreement of the two solves and phase-break encoding

For a child with two fitted incoming parents, test consecutive fully copied bins with different child labels when
at least one intervening bin has equal child labels. The passage is eligible when complementing both source indices
at either anchor reverses the ordered label pair there. In each solve, fix the parental diplotypes and the child's
copy/noncopy skeleton. Sum over four source pairs from uniform initial indices using the existing meiosis switch
rates. A transition touching a noncopy bin resets that slot to uniform and its fitted label is used in that bin.
Multiply the child's per-site genotype likelihoods, using the refit alleles, own-term-corrected flip probabilities
and cohort outlier mixture. A site with any uncalled candidate allele contributes neutral evidence to every state.
Forward and backward messages include both flanks. At the anchors restrict states to the fitted unordered pairs
and compare path mass for matching versus differing endpoint reversal bits. Add a break at the right anchor if the
log odds fall below the logit of `set_posterior`. The calculation uses independent site factors conditional on the
fixed tracks and skeleton. Per child and solve it uses O(B + m) work and O(B) additional storage with the existing
own-term correction. Independent releases use the stage's thread map.

Require genotype agreement across the two solves and retain the default solve’s calls where they agree. A heterozygote is eligible for phase when both solves determine its frame. Union break bins, then split wherever the relative flip between consecutive phased sites changes within a set. The first site is implicit. The native set preserves orientation up to one global flip.

Propagate noncopy support \(U\) separately for each solve, homologue and bin. Unlinked slots have zero support. In parent-before-child order, let \(q_0,q_1\) be the linked slot’s two conditional copy probabilities and \(U_{p0}, U_{p1}\) the corresponding parental supports. Compute \(U = (1 - (q_0+q_1)) + q_0 U_{p0} + q_1 U_{p1}\). The products of conditional marginals approximate support along the observed ancestry. A bin has high support where each solve has at least one track with \(U\) at or above `phase_posterior`, the same confidence required to release a copy source.

Within each native set and uninterrupted painting frame, a bin votes on the relative allele order only if all comparable phased heterozygotes agree. Comparable sites have two distinct, known alleles on singleton rows from the original painting panel. A high-support tract needs flanks in both immediately adjoining low-support stretches, within the same frame and set. Use the nearest determinate bin on each side. If both flanks vote alike, withhold heterozygous phase in determinate interior bins that vote oppositely. Unobserved bins still contribute to the high- and low-support runs. Painting supplies a competing interpretation of shared observations. The mask preserves ordered call arrays, genotypes, phase-set boundaries and meiosis-chain inputs.

Implementation: [science/phase/transmission_passages.py](../../haplotype_reconstruction/science/phase/transmission_passages.py), [science/phase/agreement.py](../../haplotype_reconstruction/science/phase/agreement.py), [science/phase/component.py](../../haplotype_reconstruction/science/phase/component.py).

Parameters by name: `set_posterior`, `phase_posterior`, `ORIGIN_BIN_BATCH`. The second controls only the scratch size of the parallel origin scan. See the [generated parameter reference](../reference/parameters.md).

Cost and scope: O((N + E)B) for N samples, E observed edges and B bins, plus marker scans for painting-eligible samples with high support. The mask reuses painting tie comparisons of at most O(K) per track boundary. Independent bin batches and sample releases use the stage’s budgeted thread map.

<a id="fp21"></a>
## fp21 Meiosis-chain release (set-start resets, cross-solve alignment, chain arrays)

Reset chains at component starts, near-half meiosis rates and either sample’s phase-set starts; an interior-bin start also blocks that bin. For an unlinked parent, align mirror gauge within reset segments, blocking mixed alignment. Block remaining parent-label disagreement. Test both float64 copy probabilities at the inclusive phase threshold and release a source only when aligned solves agree. Store the minimum cross-solve maximum support as float32 where released and zero elsewhere. Skipped components retain missing source and reset geometry.

Implementation: [science/phase/agreement.py](../../haplotype_reconstruction/science/phase/agreement.py), [science/phase/component.py](../../haplotype_reconstruction/science/phase/component.py), [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py).

Parameters by name: `phase_posterior`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(E B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp22"></a>
## fp22 Storage routes and release scheduling (resident/windowed, serial/pooled, spool, mapped outputs)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Choose resident or windowed likelihood storage by the declared dimension rule. Keep whole-bin orientation medians, fixed carrier-evidence chunks, identical input casts and indexed sample writes on both routes. Per-sample release may run independently, with the same fixed descendant order for both solves. Storage buffers and worker count cannot change calls, phase sets or chains.


Implementation: [science/phase/window.py](../../haplotype_reconstruction/science/phase/window.py), [store/mapped.py](../../haplotype_reconstruction/store/mapped.py).

Parameters by name: `SITE_WINDOW`, `EVIDENCE_CHUNK`, `WINDOW_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Resident evidence is O(N L); window storage bounds the live site axis while released calls and E-by-B chains remain required.

<a id="fp23"></a>
## fp23 Product record, stage identity and resume

Publish one typed PhasedSamples product per contig, binding painting, refit, sample order, breakpoints and the exact ParentEdges conditioning digest. Require matched component order and schema. Reuse only a compatible complete publication; a completion marker alone is insufficient. The stage supplies an optional storage-free `PhaseWorkspace.memo(unit, compute)` capability. For each painted component, it commits the two solved models and parents (`solve`), both solves' releases in fixed consecutive storage batches (`release/b`), then the assembled component (`component`). The memo's noncomputing `lookup(unit)` loads any stored batch subset before release work starts. All pending samples from both solves share one concurrent map and the task's CPU authority, including elastic interval threads. When the last sample in a batch finishes, `save(unit, value)` publishes that batch immediately. Batches may complete and be stored in any order; the returned releases retain canonical sample order. Storage batches never partition scheduling. Without the capability, the original computation path runs. A solve's internal sweeps remain indivisible.

Partial records live in `<contig>.ckpt.phase.partial` beside the contig checkpoint in the run's checkpoint root, independently of scratch. The ordinary container protocol publishes each complete unit atomically and recovers torn tails. Keys and identities bind the full task derivation (inputs, parameters, code and numerical policy), component, unit, batch and batch size; a different derivation is ignored. Codecs preserve exact array values and dtypes, transition logs, pedigree orders and both solves' histories. Execution callbacks and thread scratch are recreated.

`PHASE_RELEASE_BATCH = 16` in [stage parameters](../../haplotype_reconstruction/stages/phase/parameters.py) is measured in samples: fixed consecutive blocks of canonical sample order, with a shorter final batch, for both solves together. It amortises durable writes and bounds the work lost within each incomplete storage batch, independently of cores and workers. Batch size enters partial identities, with no change to scientific decisions. The task log reports the number of intermediate units reused.

The final contig record is published last. The stage completes its marker and fully decodes the final product before removing its partial container. A retry after a crash between final publication, marker creation and cleanup completes those steps without repeating science. These files are pipeline-owned restart intermediates; keeping or removing scratch does not affect their lifetime. Resident and windowed routes share the same unit identities and exact outputs.

Implementation: [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: the final product stores O(N L) calls and O(E M) chain values. The partial container additionally retains both solved model states, every release batch and the assembled component until final readback. Serialization and durability add stage work; the scientific iteration and candidate limits remain the named parameters above.

<a id="fp24"></a>
## fp24 Diagnostics and summary fields (coverage; no always-zero frames)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Report history_steps as objective-history length minus one, including repeated starting entries from later sweep calls. Preserve coverage as merged eligible painting spans, separate from chain evidence; recombination does not read it. Record settled status, pairing/junction counts, called alleles and phase sets without feeding them into another fit. Obsolete always-zero display and decoder frames are absent from the product.

Implementation: [contracts/phase.py](../../haplotype_reconstruction/contracts/phase.py), [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp25"></a>
## fp25 Full-objective enumeration and threshold reference checks

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Independent tests enumerate small block-path objectives, including blocks with children in both conventions, and compare the selected path with the full objective. Separate threshold tests check the strict block-gain and pairing-keep predicates. Production computes local row differences; it does not run a per-move full-objective verification wrapper.

Implementation: [science/phase/model.py](../../haplotype_reconstruction/science/phase/model.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.
