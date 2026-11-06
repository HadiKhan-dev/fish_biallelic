# Read evidence and quality

Evidence converts AD counts into genotype likelihoods, an observation mask and separate founder-exclusion decisions. The read model is calibrated with held-out physical folds; quality rules are empirical models, not known-truth labels. N is sample count and L site count. Inference uses no sample metadata. Read-model and quality approximation/fallback reasons remain explicit in the product.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="ec01"></a>
## ec01 VCF/BCF loader policy (record admission, duplicate positions, AD decoding, window decomposition)

Scan each indexed contig for admitted biallelic SNPs, remove all biallelic SNP records sharing a position, then decode AD on that fixed axis. Window ownership is by position, so boundaries cannot split a duplicate run. Decode-count disagreement detects a changing input. Missing or unusable AD is recorded explicitly; input FILTER does not supply site-quality decisions.

Implementation: [contracts/source.py](../../haplotype_reconstruction/contracts/source.py), [science/evidence/parameters.py](../../haplotype_reconstruction/science/evidence/parameters.py), [science/evidence/quality/parameters.py](../../haplotype_reconstruction/science/evidence/quality/parameters.py), [science/evidence/read_model/parameters.py](../../haplotype_reconstruction/science/evidence/read_model/parameters.py), [stages/inputs/vcf.py](../../haplotype_reconstruction/stages/inputs/vcf.py), [stages/inputs/sources.py](../../haplotype_reconstruction/stages/inputs/sources.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N M_in); O(M_in). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec02"></a>
## ec02 Observation-source identity (VCF path, size, mtime_ns, header order; simulation content identity)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Bind the consumed payload: positions, REF/ALT, AD, AD usability, fallback GT, contig length and loader drops. Compare that content binding before publication or resume; changed consumed content refuses reuse. Path timestamps are provenance, not evidence identity. GT changes where AD is usable do not alter the consumed payload.

Implementation: [contracts/source.py](../../haplotype_reconstruction/contracts/source.py), [stages/inputs/sources.py](../../haplotype_reconstruction/stages/inputs/sources.py), [contracts/samples.py](../../haplotype_reconstruction/contracts/samples.py), [stages/evidence/records.py](../../haplotype_reconstruction/stages/evidence/records.py), [contracts/evidence.py](../../haplotype_reconstruction/contracts/evidence.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N L) consumed-payload hashing/verification per contig, including admitted alleles and observation arrays; timestamps are not the binding.

<a id="ec03"></a>
## ec03 Canonical sample order (name sort at the source) and input-order restoration

Sort the scientific sample axis by name under the explicit run-order convention. Preserve the input axis separately for output restoration. Apply the same permutation to every per-sample array and bind both axis definitions. Names identify individuals but supply no sex, generation or parent eligibility.

Implementation: [contracts/samples.py](../../haplotype_reconstruction/contracts/samples.py), [stages/inputs/sources.py](../../haplotype_reconstruction/stages/inputs/sources.py), [core/ties.py](../../haplotype_reconstruction/core/ties.py), [stages/graph.py](../../haplotype_reconstruction/stages/graph.py), [interface/run.py](../../haplotype_reconstruction/interface/run.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N log N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec04"></a>
## ec04 Per-site genotype classification EM and clean-site leakage expectation

Compute site genotype likelihood terms, initialise the genotype-frequency mixture, and iterate E/M steps in sample order with the named floors and stopping rule. Return last-E-step moments separately from final-updated frequencies and leakage parameters. A site with no covered samples uses the explicit empty-evidence result.

Implementation: [science/evidence/quality/site_quality.py](../../haplotype_reconstruction/science/evidence/quality/site_quality.py).

Parameters by name: `classification_error`, `classification_heterozygote_alt`, `site_quality`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N M_in x iterations); O(sum of depths); O(N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec05"></a>
## ec05 Empirical-null clean core (histogram mode, half-maximum widths, local-posterior window)

Choose the empirical histogram range and width, smooth its density, select the declared tied mode, and walk/interpolate its half-maximum boundaries. Derive the clean-core/local-posterior window from that mode. The mode convention is reversible; density normalisation includes observations outside the histogram range.

Implementation: [science/evidence/quality/site_quality.py](../../haplotype_reconstruction/science/evidence/quality/site_quality.py).

Parameters by name: `core_posterior_level`, `histogram_smoothing_bins`, `site_quality`, `clip`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(M log M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec06"></a>
## ec06 r1 total-depth rule

Fit the clean total-depth window and reject only depths strictly below its lower or above its upper boundary. Equality at either boundary is retained. This rule is calibrated on the contig and is not a fixed global read-depth threshold.

Implementation: [science/evidence/quality/site_quality.py](../../haplotype_reconstruction/science/evidence/quality/site_quality.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N M_in). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec07"></a>
## ec07 r2 per-sample depth-heterogeneity rule

Sum depths over r1-passing sites before dividing to obtain sample means. Fit the clean core of positive log heterogeneity and reject only its upper tail. With no positive reference heterogeneity, retain all sites for this rule and report the reason.

Implementation: [science/evidence/quality/site_quality.py](../../haplotype_reconstruction/science/evidence/quality/site_quality.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N M_in). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec08"></a>
## ec08 r3 homozygote wrong-allele mixture (Poisson clean core vs negative-binomial artefact)

Fit clean Poisson wrong-allele counts against a negative-binomial artefact class on the eligible sites. Preserve the fixed chunk reduction, initialisation, constrained Newton/bracketing steps and mixture stopping rules. Apply the recorded clean-versus-artefact comparison; insufficient or boundary information follows the declared fit branch, not an ad hoc error-rate substitute.

Implementation: [science/evidence/quality/site_quality.py](../../haplotype_reconstruction/science/evidence/quality/site_quality.py).

Parameters by name: `clean_posterior_level`, `minimum_artefact_separation`, `site_quality`, `clip`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec09"></a>
## ec09 r4 heterozygote-balance two-beta-binomial mixture, labelling, gate and abstention

On sites passing r1–r3, fit the two-component beta-binomial balance mixture from the declared starts in canonical histogram order. Polish candidates, label the clean component, and apply fit/admission/abstention gates before rejection. The best-start ordering is fixed and is not part of convention reversal.

Implementation: [science/evidence/quality/site_quality.py](../../haplotype_reconstruction/science/evidence/quality/site_quality.py).

Parameters by name: `minimum_artefact_separation`, `clean_posterior_level`, `site_quality`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec10"></a>
## ec10 r5 informativeness (expected carriers), recounted among unflagged samples

Reject sites whose expected non-major-genotype carrier count is strictly below the informativeness threshold. If sample QC flags libraries, rerun the genotype calculation on unflagged samples and replace r5 only. r1–r4 are not recalibrated. The drop-uninformative setting decides whether the recorded r5 bit removes a site.

Implementation: [science/evidence/quality/site_quality.py](../../haplotype_reconstruction/science/evidence/quality/site_quality.py).

Parameters by name: `drop_uninformative_sites`, `minimum_expected_carriers`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: An additional site-mixture fit on unflagged samples when the rule requires it.

<a id="ec11"></a>
## ec11 Site-filter applicability, rule combination, rule bits and record

If filtering is disabled or calibration is inapplicable, retain all sites and record why. Otherwise execute r1, r2, r3, r4, initial r5, sample QC, then the r5 recount. OR only applied rules to form the kept mask; preserve every rule bit and distinguish singly removed sites.

Implementation: [science/evidence/run.py](../../haplotype_reconstruction/science/evidence/run.py), [stages/evidence/tasks.py](../../haplotype_reconstruction/stages/evidence/tasks.py).

Parameters by name: `enabled`, `minimum_calibration_sites`, `site_quality`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N M_in); O(M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec12"></a>
## ec12 Sample QC: carriers, eligible cells, composite fragment weights and regions

Construct carrier thresholds from the clean error law, identify eligible/minor cells and assign complete fragment windows to regions. Weight each cell by its fragment’s complete cell count, including windows spanning processing tiles. Aggregate exact histogram codes; a tile boundary cannot divide the biological fragment weight.

Implementation: [science/evidence/quality/sample_quality.py](../../haplotype_reconstruction/science/evidence/quality/sample_quality.py).

Parameters by name: `carrier_tail_level`, `fragment_bp`, `artefact_regions`, `initial`, `sample_quality`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N M_pass). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec13"></a>
## ec13 Sample QC: per-library null/alternative mixture fits with regional artefact shares

Fit a per-library null mixture, then warm-start alternatives in ascending foreign-fraction order. Retain the strictly best alternative’s weights and polish the null from them. Regional artefact shares remain part of the mixture. The warm-start path and pivot rule are model decisions, not interchangeable optimiser conveniences.

Implementation: [science/evidence/quality/sample_quality.py](../../haplotype_reconstruction/science/evidence/quality/sample_quality.py).

Parameters by name: `minimum_foreign_fraction`, `heterozygote_floor_ratio`, `heterozygote_depth_classes`, `sample_quality`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(G I (B Q^2 + (J Q)^3 + R_B S_b)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec14"></a>
## ec14 Sample QC: statistic, working threshold, applicability guards and flags

Check minimum sample/cell and fraction-separation applicability first. Profile finite alternatives against the null, take the fixed first maximum, and form the nonnegative likelihood-ratio statistic. Flag a library inclusively at the working threshold. Record inapplicability; flagged libraries affect r5 and founder construction, not their own later phasing.

Implementation: [science/evidence/quality/sample_quality.py](../../haplotype_reconstruction/science/evidence/quality/sample_quality.py).

Parameters by name: `minimum_samples`, `carrier_tail_level`, `enabled`, `sample_quality`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec15"></a>
## ec15 Sample QC: inherited-segment report

Identify paired near-private carriers, classify heterozygote-like cells, then sort by sample, partner and position. Split fragment events and inherited regions at their named gaps; report regions meeting the event threshold for unflagged samples. Cell counts/read fractions include the complete pair region. This is a diagnostic, not a parent call.

Implementation: [science/evidence/quality/sample_quality.py](../../haplotype_reconstruction/science/evidence/quality/sample_quality.py).

Parameters by name: `segment_gap_bp`, `segment_minimum_events`, `fragment_bp`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(near-private cells log). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec16"></a>
## ec16 Read-model emission family (binomial homozygotes, beta-binomial heterozygote, prefix sums)

Evaluate binomial homozygote and beta-binomial heterozygote emissions using common prefix terms. Reduce sample contributions in the declared order without reassociation. The concentration parameter models heterozygote overdispersion; genotype mixtures and model selection are separate operations.

Implementation: [science/evidence/read_model/emission.py](../../haplotype_reconstruction/science/evidence/read_model/emission.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(C + D); O(N C). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec17"></a>
## ec17 Calibration histograms and physical folds

Admit calibration cells within the declared depth range, count depth/ALT histograms per sample, and assign physical-position folds. Preserve complete-cell counts and fixed fold geometry. Training and held-out evidence stay separate; computational tiles do not define statistical folds.

Implementation: [science/evidence/read_model/calibrate.py](../../haplotype_reconstruction/science/evidence/read_model/calibrate.py).

Parameters by name: `maximum_fit_depth`, `fold_window_bp`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec18"></a>
## ec18 Shared-binomial fit: EM starts, stationary polish, endpoint equivalence and representative

Sort sample histograms by content, fit all declared shared-binomial starts, and enforce monotonicity, finiteness and stationarity. Polish and compare endpoints with the declared equivalence/representative rule, then restore the sample axis. Distinct unresolved endpoints cause a recorded fallback reason; optimiser success alone is not the decision.

Implementation: [science/evidence/read_model/fit.py](../../haplotype_reconstruction/science/evidence/read_model/fit.py).

Parameters by name: `em_starts`, `em_maximum_iterations`, `em_tolerance_per_observation`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N C). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec19"></a>
## ec19 Pooled sample binomial (diagonal-Laplace empirical Bayes)

Fit sample-local binomial coordinates, estimate diagonal curvature/variance, and fit pooled hyperparameters from finite informative samples. Use the shared value when variance information is insufficient and a point mass when the between-sample variance is at its boundary. Refit with the empirical-Bayes prior; weak curvature is not treated as precise evidence.

Implementation: [science/evidence/read_model/fit.py](../../haplotype_reconstruction/science/evidence/read_model/fit.py).

Parameters by name: `polish`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(C). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec20"></a>
## ec20 Joint sample beta-binomial (shared heterozygote concentration)

Start from pooled local effects and jointly fit a shared heterozygote concentration with sample effects. Preserve bound handling and the arrowhead-Hessian Newton polish. Optimiser success is diagnostic; held-out predictive selection decides whether this more complex model is used.

Implementation: [science/evidence/read_model/fit.py](../../haplotype_reconstruction/science/evidence/read_model/fit.py).

Parameters by name: `initial_heterozygote_concentration`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N C). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec21"></a>
## ec21 Held-out stretch scores and sequential nested model selection

Sum held-out log-predictive gains by physical stretch in sample order, discard unexposed stretches, then compare nested candidates sequentially. Require enough stretches, positive total gain and the inclusive standardised-gain threshold. Non-finite complex candidates are skipped; an unusable baseline invokes the fixed fallback.

Implementation: [science/evidence/run.py](../../haplotype_reconstruction/science/evidence/run.py), [science/evidence/read_model/calibrate.py](../../haplotype_reconstruction/science/evidence/read_model/calibrate.py).

Parameters by name: `selection_z`, `selection_minimum_blocks`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L); O(M C N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec22"></a>
## ec22 Fixed binomial fallback and fallback reasons

Use the fixed binomial model when calibration is disabled, folds lack admitted evidence, the shared fit is unresolved/failed, or its predictive total is non-finite. A failed complex fit alone does not discard a valid simpler candidate. A joint-fit exception keeps the fitted sample-binomial candidate in held-out selection; a pooled-fit exception leaves only the shared binomial candidate. Record either exception as complex_fit_failure. Return a typed fallback reason for the application layer to report.

Implementation: [science/evidence/read_model/calibrate.py](../../haplotype_reconstruction/science/evidence/read_model/calibrate.py), [science/evidence/read_model/fit.py](../../haplotype_reconstruction/science/evidence/read_model/fit.py), [stages/evidence/tasks.py](../../haplotype_reconstruction/stages/evidence/tasks.py), [science/evidence/run.py](../../haplotype_reconstruction/science/evidence/run.py).

Parameters by name: `fixed_error`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Constant rule construction plus the represented diagnostic fields.

<a id="ec23"></a>
## ec23 Genotype-likelihood application, validation and the observation mask

Build per-sample prefix tables and a bounded-depth lookup; evaluate deeper cells with the same expression. Require zero-depth likelihoods to be exactly neutral and observed likelihoods finite, nonnegative and normalised. The lookup depth changes cost, not the emission model or observation mask.

GL storage is chosen automatically for each sample and contig. Under the fixed/shared binomial, sample-binomial and sample-joint models, the fitted error and heterozygote parameters are fixed within a sample/contig. A site's normalised triple therefore depends only on its integer REF/ALT depth pair. There are no site-specific emission parameters or genotype priors; `(0, 0)` has the literal float64 `1/3` triple. Storage fills triples from the unchanged dense kernel at their first-occurrence sites, including the existing lookup/direct path, and checks them independently against bounded windows of original sites.

For `D` distinct depth pairs and `L` sites, choose the smallest unsigned index type whose zero-based codes represent `D - 1`: uint8 through 256 pairs, uint16 through 65,536, then uint32 through 4,294,967,296. The automatic byte-cost rule strictly requires `32*D + width*L + 17 < 33*L`. A dense cell contains three float64 GLs (24 bytes), two int32 counts (8 bytes) and a bool mask (1 byte). A table entry is the exact float64 triple and int32 pair (32 bytes); maps add 17 bytes per sample and one final eight-byte offset per contig. Actual aggregate container bytes must also beat the joint dense total. There is no user setting, depth cap, saving-margin requirement or construction-admission threshold.

Each coded pair recovers its counts exactly; `(0,0)` is unobserved. The same code stream and pair table therefore serve GL, counts and flags. Dense fallback rows alone retain separate count and mask arrays. `PairArray` borrows the likelihood source and only its own fallback field. A founder mask adds excluded-row flags without copying a contig mask or changing raw counts/GLs.

Planning retains only distinct-pair cardinalities and widths across samples. Safety checks and final filling rebuild one sample's integer key/first-site plan at a time, preventing N simultaneous plans or duplicated float64 tables. Every proposed codebook is checked against all source GL bits in bounded windows. A repeated pair with different triples forces dense storage. Dense fallback uses the original dense kernel and output allocator. Site/sample QC, model calibration and discovery-depth classification still consume their dense input counts before final encoding; this changes storage, not their kernels or decisions.

Explicit gathers materialise identical ordered GL, count and mask slices. Discovery block/window declarations are lazy for both GL and mask. Relatedness frequency and pedigree availability use bounded site gathers; painting uses checkpoint windows. Assembly, pedigree exclusion, refit, phase and legacy painting-report APIs keep consumer-local dense projections where the existing boundary requires them. No full coded count/mask rows are stored alongside their codebook.


Implementation: [science/evidence/likelihoods.py](../../haplotype_reconstruction/science/evidence/likelihoods.py), [science/evidence/run.py](../../haplotype_reconstruction/science/evidence/run.py), [science/evidence/read_model/emission.py](../../haplotype_reconstruction/science/evidence/read_model/emission.py), [stages/evidence/tasks.py](../../haplotype_reconstruction/stages/evidence/tasks.py).

Parameters by name: `lookup_depth`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: likelihood cell evaluation is O(N L) plus per-sample depth-prefix/lookup work. Joint encoding sorts L integer keys per sample and plans/checks/fills codes with searchsorted: O(N L log L + sum_n L log D_n) work, where D_n <= L is that sample's distinct-pair count. Planning scratch is O(L + D_n) for one sample at a time; final joint storage must be strictly smaller than 33 N L bytes. GL/mask projections use an 8 MiB workspace target, at most 2048 sites per tile, with at least one site even when the sample axis alone exceeds that target. Tile size is independent of GL-only storage savings.

<a id="ec24"></a>
## ec24 Discovery sites and block partition

Use exact integer depth sums and the inclusive discovery-site threshold. Partition the contig into fixed site-count blocks independently of genomic spacing and the selected discovery mask. This evidence partition supplies the local searches; it is not an adaptive pedigree segmentation.

Implementation: [science/evidence/blocks.py](../../haplotype_reconstruction/science/evidence/blocks.py), [science/evidence/run.py](../../haplotype_reconstruction/science/evidence/run.py), [stages/evidence/tasks.py](../../haplotype_reconstruction/stages/evidence/tasks.py), [science/evidence/parameters.py](../../haplotype_reconstruction/science/evidence/parameters.py).

Parameters by name: `minimum_total_depth`, `block_sites`, `discovery_sites`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec25"></a>
## ec25 Founder view (QC-flagged samples as missing for founder construction)

Expose a founder-construction view in which QC-flagged samples are neutral/missing. Keep the original evidence for downstream sample inference. Exclusion therefore changes who supports founder estimation without deleting an individual from the run.

Implementation: [stages/evidence/records.py](../../haplotype_reconstruction/stages/evidence/records.py), [contracts/evidence.py](../../haplotype_reconstruction/contracts/evidence.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N L) observation-mask storage per founder view when exclusions require a private mask; evidence arrays remain shared.

<a id="ec26"></a>
## ec26 Evidence record, report, identity, resume and published QC tables

Derive the evidence identity from the consumed source and scientific parameters, then reuse a compatible checked product or compute and publish a new typed product. Store observations, quality/read-model decisions and diagnostics with explicit roles. Report tasks render QC separately; timers do not enter identity.

New evidence publications use schema 2. Non-observation fields and decisions/diagnostics are unchanged. A dense contig stores `gl_encoding="dense"` and its original arrays. The accepted GL-only v2 layout (`gl_encoding="coded"`) remains readable. New joint publications use `gl_encoding="paired"`, flat numeric `gl_*` fields, `counts_dense[Ndense,L,2]` and `observed_dense[Ndense,L]`; whole-contig `counts` and `observed` fields are omitted. Existing numeric checkpoint and worker transport formats carry these arrays without expansion or another copy of the shared code stream.

The adapter reads schema 1, dense/GL-only schema 2, and joint schema 2 under an explicitly expected derivation. `load` preserves resident GL and mask sources and includes a count source only when requested. The mask never retains dense fallback counts. `open_arrays` exposes the three logical names over shared physical code/pair handles and only the dictionary rows requested by each window. `load_arrays` and `load_dense` stream explicit dense projections directly into one final destination, including caller-owned destinations. For a sample window of W sites, dictionary lookup sorts/deduplicates W codes (O(W log W) work, O(W) scratch), decodes at most min(D, W) rows in consecutive runs, and restores the requested order. Interleaved sample batches do not reread whole D-row dictionaries. Over a complete pass, requested dictionary bytes are O(N L), with O(N L log W) lookup work; physical compressed-chunk reads additionally depend on the store's bounded shared cache and code locality. Identity validation remains separate from format compatibility; public readers rebuild current derivations, so schema-1 format support is not transparent 4.3 resume compatibility (see the [migration guide](../user/migration.md#migrating-from-43-to-44)).


Implementation: [stages/evidence/records.py](../../haplotype_reconstruction/stages/evidence/records.py), [contracts/evidence.py](../../haplotype_reconstruction/contracts/evidence.py), [stages/evidence/tasks.py](../../haplotype_reconstruction/stages/evidence/tasks.py), [science/evidence/run.py](../../haplotype_reconstruction/science/evidence/run.py), [outputs/reports/evidence.py](../../haplotype_reconstruction/outputs/reports/evidence.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(M_in). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<!-- named mechanism anchors -->

## Named mechanisms

<a id="records"></a>

**Records:** see [EC26](#ec26).

<a id="sample-order"></a>

**Sample order:** see [EC01](#ec01).

<a id="sample-quality"></a>

**Sample quality:** see [EC12](#ec12).

<a id="stage"></a>

**Stage:** see [EC26](#ec26).
