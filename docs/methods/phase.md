# Family phase

Family phase conditions on frozen painting, refit alleles and released parent edges. Two convention solves optimise a founder-mosaic objective; release requires their agreement while preserving uncertainty and phase-set boundaries. N is samples, B bins, L sites, S labels and E parent edges. The modular conditional model does not refit upstream ancestry or pedigree. A converged coordinate ascent need not be a global optimum.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="fp01"></a>
## fp01 Component model inputs: bins, labels and tempered pair emissions (cache or recompute)

Build symmetric pair emissions from the painting record’s own robustness, floor and site-correlation parameters. Use its packed cache or recompute the same binned emissions and design effect, including empty-information bins. Require painting/pedigree sample alignment. Both solves share these immutable pre-refit emissions; later allele refit does not change this ancestry objective.

Implementation: [contracts/phase.py](../../haplotype_reconstruction/contracts/phase.py), [science/painting/emissions.py](../../haplotype_reconstruction/science/painting/emissions.py), [science/phase/component.py](../../haplotype_reconstruction/science/phase/component.py), [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py), [science/phase/painting_inputs.py](../../haplotype_reconstruction/science/phase/painting_inputs.py), [stages/phase/records.py](../../haplotype_reconstruction/stages/phase/records.py).

Parameters by name: `SITE_WINDOW`, `robustness_epsilon`, `emission_log_floor`, `emission_site_correlation`, `EMPTY_COMPONENT_BIN_SITES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N S^2 B); O(N S^2 L_c). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp02"></a>
## fp02 Edge rates and initial terms (label switch, Haldane meiosis switch with protected resets, not-a-copy entry/exit)

Derive label-switch rates from recorded painting transitions, meiosis-switch rates from map distances and the shared Haldane fraction, and not-a-copy entry/exit rates from physical distance. Protected intervals force meiosis resets. L1 unifies only the Haldane formula; the distinct not-a-copy expressions retain their declared arithmetic. Initialise linked slots with the stationary copy/not-copy terms and preserve the reset boundary test.

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
## fp08 D2 parents' pairing joint moves, solve loop and pairing ambiguity

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

Parameters by name: `junction_min_samples`, `junction_pin_nats`, `junction_pin_bins`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

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

<a id="fp14"></a>
## fp14 CAL2 panel-allele uncertainty (prior, carrier evidence, flips, own terms)

Form the chromosome-wide allele prior from all called refit cells. Accumulate robust carrier evidence from solve-MAP labels, excluding founder-excluded samples, in fixed site chunks and homologue-then-sample order. Convert it to capped allele-flip uncertainty; unknown alleles have no flip correction. Remove the released sample’s own terms by subtraction before its site model. Resident/windowed routes preserve the same float32 evidence inputs and float64 accumulation.

Implementation: [science/phase/allele_evidence.py](../../haplotype_reconstruction/science/phase/allele_evidence.py).

Parameters by name: `FLIP_NEGLIGIBLE`, `FLIP_CEILING`, `PRIOR_BOUNDS`, `EVIDENCE_CHUNK`, `clip`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N m); O(m); O(S m). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp15"></a>
## fp15 Block posterior (forward-backward), label-pair posterior and chain marginals

Compute the exact conditional block forward/backward posterior with gauge starting at zero and any anchor mask enforced in both directions. Sum gauge and other-slot states in declared row order, applying the small-log-mass cutoff without renormalisation. L2 uses topological depth/ascending-index descendant order for release in both solves, independent of workers. L4 retains chain probabilities as float64 until release tests, casting only stored support.

Implementation: [science/phase/kernels.py](../../haplotype_reconstruction/science/phase/kernels.py), [science/phase/release_kernels.py](../../haplotype_reconstruction/science/phase/release_kernels.py).

Parameters by name: `LABEL_SWITCH_MINIMUM`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(B (S + 2)^3 G). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp16"></a>
## fp16 Site model (ordered allele posteriors per site)

Mix known label-pair mass with CAL2 allele uncertainty and the site-deviation law. Preserve exact-zero grouping: unchanged pairs, one-flipped-label sides and individually normalised two-flipped pairs are accumulated in their specified order. Skip unknown pairs or nonpositive normalisers. Validate posterior settings against the model’s attainable certainty ceiling. Sites are independent conditional on the block posterior, but the block remains pedigree conditioned.

Implementation: [science/phase/kernels.py](../../haplotype_reconstruction/science/phase/kernels.py), [science/phase/release_kernels.py](../../haplotype_reconstruction/science/phase/release_kernels.py).

Parameters by name: `site_deviation`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(m S^2); O(m (4 + 4F + F^2)); O(K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp17"></a>
## fp17 Genotype and per-site phase calls

Call genotype from the unanchored genotype posterior, with the lower genotype winning an exact maximum tie and an inclusive call threshold. Use the ordered posterior to phase called heterozygotes at its own inclusive threshold. A strict P10 advantage gives reversed allele order; otherwise retain P01 order even for an unphased heterozygote. Uncalled sites remain missing and frame uncertainty is explicit.

Implementation: [science/phase/release.py](../../haplotype_reconstruction/science/phase/release.py).

Parameters by name: `genotype_posterior`, `phase_posterior`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(m). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp18"></a>
## fp18 Gauge-sample anchor partitions and orientation-certainty set ends

For an unlinked sample without child tracks, partition at hard cuts and choose the first confident heterozygous anchor in each interval. Anchor by strict label order and recompute the conditional posterior. End the set at the first later sufficiently heterozygous bin losing orientation certainty; NaN certainty does not trigger that stop. Include pre-anchor bins in the anchored partition and compute orientation medians over whole bins. A partition without an anchor releases no phase.

Implementation: [science/phase/release.py](../../haplotype_reconstruction/science/phase/release.py).

Parameters by name: `phase_posterior`, `genotype_posterior`, `set_posterior`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(m_partition S^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp19"></a>
## fp19 Children-anchored painting-tie gate (B)

For an unlinked sample with children, inspect finite orientation bins immediately before and at/after each tie-only cut. Bridge only when both sides reach the set-posterior threshold. Each solve decides separately and agreement keeps any break either retains. Linked samples and gauge-only samples do not use this gate.

Implementation: [science/phase/release.py](../../haplotype_reconstruction/science/phase/release.py).

Parameters by name: `set_posterior`, `genotype_posterior`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(m S^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp20"></a>
## fp20 Agreement of the two solves and phase-break encoding

Require genotype agreement across the two solves and retain the default solve’s calls where they agree. A heterozygote is phased only when neither solve leaves its frame undetermined. Union break bins, then split again wherever the relative flip between consecutive phased sites changes within a set. The first site is not an explicit break. Release preserves orientation only up to a single flip per resulting phase set.

Implementation: [science/phase/agreement.py](../../haplotype_reconstruction/science/phase/agreement.py), [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(m). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp21"></a>
## fp21 Meiosis-chain release (set-start resets, cross-solve alignment, chain arrays)

Reset chains at component starts, near-half meiosis rates and either sample’s phase-set starts; an interior-bin start also blocks that bin. For an unlinked parent, align mirror gauge within reset segments, blocking mixed alignment. Block remaining parent-label disagreement. Test both float64 copy probabilities at the inclusive phase threshold and release a source only when aligned solves agree. Store the minimum cross-solve maximum support as float32 where released and zero elsewhere. Skipped components retain missing source and reset geometry.

Implementation: [science/phase/agreement.py](../../haplotype_reconstruction/science/phase/agreement.py), [science/phase/component.py](../../haplotype_reconstruction/science/phase/component.py), [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py).

Parameters by name: `phase_posterior`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(E B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp22"></a>
## fp22 Storage routes and release scheduling (resident/windowed, serial/pooled, spool, mapped outputs)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Choose resident or windowed likelihood storage by the declared dimension rule. Keep whole-bin orientation medians, fixed CAL2 chunks, identical input casts and indexed sample writes on both routes. Per-sample release may run independently, but L2 fixes its descendant order for both solves. Storage buffers and worker count cannot change calls, phase sets or chains.

Implementation: [science/phase/window.py](../../haplotype_reconstruction/science/phase/window.py), [store/mapped.py](../../haplotype_reconstruction/store/mapped.py).

Parameters by name: `SITE_WINDOW`, `EVIDENCE_CHUNK`, `WINDOW_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Resident evidence is O(N L); window storage bounds the live site axis while released calls and E-by-B chains remain required.

<a id="fp23"></a>
## fp23 Product record, stage identity and resume

Publish one typed PhasedSamples product per contig, binding painting, refit, sample order, breakpoints and the exact ParentEdges conditioning digest. Require matched component order and schema. Reuse only a compatible complete publication; a completion marker alone is insufficient. Only final published products resume—intermediate family-phase iterations are not durable restart states.

Implementation: [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N L); O(E M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp24"></a>
## fp24 Diagnostics, summary and legacy fields (coverage, zero frames)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Report history_steps as objective-history length minus one, including repeated starting entries from later sweep calls. Preserve coverage as merged eligible painting spans, separate from chain evidence; recombination does not read it. Record settled status, pairing/junction counts, called alleles and phase sets without feeding them into another fit. Obsolete always-zero display and decoder frames are absent from the product.

Implementation: [contracts/phase.py](../../haplotype_reconstruction/contracts/phase.py), [stages/phase/tasks.py](../../haplotype_reconstruction/stages/phase/tasks.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fp25"></a>
## fp25 Full-objective enumeration and threshold reference checks

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Independent tests enumerate small block-path objectives, including blocks with children in both conventions, and compare the selected path with the full objective. Separate threshold tests check the strict block-gain and pairing-keep predicates. Production computes local row differences and records diagnostic traces; it does not run a per-move full-objective verification wrapper.

Implementation: [science/phase/model.py](../../haplotype_reconstruction/science/phase/model.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N B). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.
