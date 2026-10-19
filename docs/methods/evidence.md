# Evidence

The evidence stage turns one contig's allele depths into the observations every later stage reads:
the site axis after quality control, the REF/ALT read counts, normalised genotype likelihoods under
a read model calibrated on the contig's own reads, the observation mask, the sites the discovery
search uses, per-sample contamination flags and the partition of the site axis into discovery
blocks. It is the first stage of the pipeline and runs once per contig; nothing downstream
re-normalises, re-filters or re-validates these observations.

## Inputs and outputs

| | |
| --- | --- |
| input | `ContigReads` (`model/evidence.py`): sample ids, positions (1-based VCF POS, strictly increasing) and REF/ALT read counts `int32[N, L, 2]` of the contig's biallelic SNPs, from the VCF/BCF loader (`evidence/vcf.py`) or the simulation source (`simulation/source.py`, nothing dropped) |
| product | contig record `evidence`: a `ContigEvidence` (positions, counts, likelihoods `float64[N, L, 3]`, observed `bool[N, L]`, discovery sites `bool[L]`, founder exclusions `bool[N]`, block spans `int64[B, 2]`) and the stage report (loader counts, site and sample QC, read-model fits and scores) |
| identity | `identity/evidence` per contig: the evidence configuration, the observation source and the code of the stage |
| published | `quality/<contig>.samples.tsv` and `quality/<contig>.segments.tsv` (contamination statistics and inherited-segment regions; only where the sample QC ran) |

User options (see [configuration](../configuration.md)): `[quality] site_filter` (`auto` = on for
VCF input, off for simulations), `[quality] sample_filter` (`auto` likewise; it runs only where the
site filter is applied) and `[quality] read_calibration` (off = the fixed binomial model). Every
other setting is an internal constant of `EvidenceConfig` (`evidence/config.py`).

## Missingness conventions

Genotypes are counted in ALT alleles (g = 0, 1, 2); founder alleles downstream are 0 (REF), 1 (ALT)
or -1 (unknown). The genotype likelihoods and the observation mask are separate quantities: a
cell without reads has the uniform likelihood (1/3, 1/3, 1/3) and is unobserved, so a flat
likelihood at a missing cell is never read as an observed heterozygote. Missing data never counts
as an allele match; the founder-construction stages give unobserved cells a neutral emission.

## Loader policy

One policy holds for every VCF/BCF input (`evidence/vcf.py`). The file must be indexed and declare
FORMAT/AD; GT- or PL-only input is refused, because genotype calls or likelihoods do not identify
the read-level parameters the read model estimates. Per record:

- a record with more than one ALT allele is dropped as `multiallelic` (its depths are never
  truncated into biallelic ones);
- a record whose REF or ALT is not one of A, C, G, T (indels, MNPs, symbolic alleles, `*`, no ALT)
  is dropped as `not_snp`;
- when two or more biallelic SNP records share a position, all of them are dropped as
  `duplicate_position`, because the file does not decide which one is the site;
- a missing AD value, or a record without AD, is zero reads (an unobserved cell).

The dropped counts are reported per reason in the evidence report and the log. Everything else
that is not as expected is an error: unreadable files, unknown contigs or samples, records out of
position order, and AD fields that do not hold one REF and one ALT depth per sample. A contig with
no biallelic SNP, or with no site left after the site filter, is an error too.

## Site quality filter

With `[quality] site_filter` on (the default for VCF input), each contig's SNPs pass a
self-calibrated filter (`evidence/site_quality.py`) before read calibration and discovery. It is
calibrated on the contig's own depth distributions only: no sample labels, pedigree, benchmark or
truth enter. With d_sl the depth of sample s at site l and D_l = sum_s d_sl, a site is removed
when any of these rules fires:

| rule | statistic | removed when |
| --- | --- | --- |
| r1 | total depth D_l | outside the clean-core window of the total-depth histogram (both tails) |
| r2 | depth heterogeneity X_l = sum_s (d_sl - E_sl)^2 / E_sl / (N - 1), with E_sl = D_l m_s / sum_t m_t and m_s sample s's mean depth over r1-passing sites | above the clean-core window of log X over r1-passing sites |
| r3 | wrong-allele reads in samples classified homozygous | a negative-binomial artefact class explains them better than the Poisson clean core |
| r4 | ALT fraction in samples classified heterozygous | a beta-binomial artefact class explains it better than the binomial clean core |
| r5 | expected carriers of a non-major genotype, c_l (1 - max_g F_lg) | below `EvidenceConfig.site_quality.minimum_expected_carriers = 2.0` |

**Clean cores (r1, r2).** The statistic's histogram (Freedman-Diaconis bin width over its 0.5-99.5
percentile range; integer bins for depth) is smoothed by a moving average over
`EvidenceConfig.site_quality.histogram_smoothing_bins = 5` bins. The clean core is the histogram
mode with Gaussian-equivalent left and right standard deviations from the half-maximum
crossings, its density anchored at the peak (an empirical null). The window is the run of bins
around the mode where the core's local posterior, null density over observed density, is at
least `EvidenceConfig.site_quality.core_posterior_level = 0.5`.

**Genotype classification (r3-r5).** Each site's genotypes are classified under a fixed read model,
ALT reads ~ Binomial(d, p_g) with p = (eps, h, 1 - eps),
`EvidenceConfig.site_quality.classification_error = 0.01` and
`EvidenceConfig.site_quality.classification_heterozygote_alt = 0.5`, by EM over the site's free
genotype frequencies F_l (no Hardy-Weinberg constraint). The E-step gives the expected
wrong-allele reads W_l and depth N_l in homozygotes, the expected ALT reads A_l and depth H_l in
heterozygotes, and c_l, the number of samples with reads. The clean-core rates are then fitted,
not fixed:

- r3, among r1/r2-passing sites with N_l >= 1: clean W ~ Poisson(N eps0), artefact
  W ~ NegBin(mean N eps1, shape k), fitted by EM with numerical M-steps;
- r4, among r1-r3-passing sites with H_l >= 1: clean A ~ Binomial(H, h0), artefact
  A ~ beta-binomial(mean m, concentration c). The clean heterozygote ALT fraction h0 is also the
  balance the sample QC uses.

A site is flagged by r3 or r4 when its posterior probability of the clean class is below
`EvidenceConfig.site_quality.clean_posterior_level = 0.5`, and only when the artefact class
separates from the clean core: eps1 >= 3 eps0 for r3; for r4 an artefact standard deviation, or a
mean offset |m - h0|, of at least 3 binomial standard deviations at the median heterozygote depth
(`EvidenceConfig.site_quality.minimum_artefact_separation = 3.0`). A mixture whose artefact class
does not separate removes nothing.

Rule r5 is informativeness, not quality: it removes sites without segregating information
(`EvidenceConfig.site_quality.drop_uninformative_sites = True`). When the sample QC flags samples on
the contig, r5 counts carriers among the unflagged samples only.

The filter runs only on contigs with at least
`EvidenceConfig.site_quality.minimum_calibration_sites = 1000` SNPs; on shorter contigs every site
is kept and the reason is recorded. The report keeps, per input SNP, a bit per rule that fired,
the per-rule counts, the counts removed by one rule alone and the fitted cores and mixtures.

The filter is off by default for simulations: the depth rules cannot tell benign sample-specific
missingness (for example simulated dropout tracts) from mapping artefacts; when the read error rate
is several times the 1% classification error, homozygotes with error reads are classified
heterozygous and r4 separates spuriously; and r5 removes founder-monomorphic simulated sites.

## Sample quality

Inside the site filter, each contig checks every analysed sample for library contamination
(`evidence/sample_quality.py`). Foreign DNA at read fraction alpha < 1/2 shows alleles that no
other sample carries, on every contig, at a minor-read fraction near alpha (foreign homozygote) or
alpha/2 (foreign heterozygote). A sample's own heterozygous allele that no other sample carries (an
inherited segment absent from the sequenced founders, or a founder segment no sequenced descendant
received) shows the heterozygote balance instead, about one half, and is regional. Only the first
is flagged.

At the SNPs that pass r1-r4 (whole analysed cohort), the cohort-minor allele is ALT when the
cohort's ALT reads do not exceed its REF reads; a sample's cell carries it with at least one minor
read. Per sample on one contig:

- **Private cells** are those where the sample is the only carrier. Consecutive private cells less
  than `EvidenceConfig.sample_quality.fragment_bp = 1000` bp apart can come from the same sequenced
  fragments and form one event; each cell weighs 1 / (event size).
- **Mixture.** The minor reads k of depth d of each private cell follow a mixture of a heterozygous
  allele (fraction h = h0 from r4, or 1 - h0 when the minor allele is REF) and one low-fraction
  allele (fraction a, fitted per sample in (0, h/2]: h/2 is the largest fraction of a foreign
  heterozygous allele while foreign DNA is under half the library), both binomial and truncated at
  k >= 1. The mixing weight (exactly, for the current a) and a are fitted by coordinate ascent; the
  low-fraction events are the event-weighted posterior mass of the low component at that fit, so
  heterozygous cells, however many, are not counted.
- **Statistic.** sqrt(low-fraction events / observed cells), the variance-stabilising scale of an
  event count. The sample's robust z is taken against the cohort on that contig: median and
  1.4826 x MAD over samples with observed cells, the spread never smaller than the statistic's
  counting noise 0.5 / sqrt(median observed cells).
- **Flag.** A sample is flagged on the contig when
  z > `EvidenceConfig.sample_quality.flag_robust_z = 5.0`. With fewer than `EvidenceConfig.sample_quality.minimum_samples = 20` samples with observed cells
  nothing is flagged and the reason is recorded.

No labels, pedigree, benchmark or absolute rate enter. The decision is per contig, so a contig's
result does not depend on which other contigs a run includes. The QC runs only where the site filter
is applied.

**Inherited segments (reported, never flagged).** Near-private cells (the sample and at most one
partner carry the allele) whose reads favour the heterozygous fraction over h/2 are grouped per
(sample, partner) into fragment events and into regions of events less than
`EvidenceConfig.sample_quality.segment_gap_bp = 1000000` bp apart. Regions with at least
`EvidenceConfig.sample_quality.segment_minimum_events = 3.0` events are listed for unflagged samples
with the region's near-private cell count and the sample's and partner's minor-read fractions.
Such segments, shared by a pair of samples or private to one, can explain added founder rows and
pedigree misfits; they are not evidence of contamination.

**Effect of a flag.** On the contig where it is flagged, a sample leaves founder construction:
`ContigEvidence.founder_view()` gives it zero counts, uniform likelihoods and no observation (the
state of a zero-depth cell), and discovery, local selection, assembly, founder refinement and the
founder refit read that view. The site axis, the discovery sites and the blocks are unchanged (they
are computed from all samples). Read calibration, painting, the pedigree, family phase and
recombination use every sample's evidence, so flagged samples are still painted, phased and scored.

## Read-model calibration

The read model (`evidence/read_model/`) gives each analysed sample s three genotype kernels for a
cell with d reads of which a are ALT (the binomial coefficient is common to all three and cancels):

- g = 0 (REF/REF): binomial with ALT probability e_s;
- g = 1 (REF/ALT): beta-binomial with mean q_s = e_s + (1 - 2 e_s) b_s and heterozygote dispersion;
- g = 2 (ALT/ALT): binomial with ALT probability 1 - e_s.

Here e_s is the sample's homozygote error, b_s its allele balance (so e_s < q_s < 1 - e_s), and the
heterozygote dispersion is 1/kappa for one contig-shared concentration kappa; zero dispersion is the
binomial limit, evaluated exactly. Each sample also has an unconstrained three-genotype mixture
for fitting and predictive scoring; these weights are not Hardy-Weinberg frequencies and are never
applied as genotype priors.

### Candidate models

The candidates are nested, from simplest (their order is the tie rule):

1. `binomial`: one shared e and q, fitted by EM from the starts
   `EvidenceConfig.read_model.em_starts = ((0.02, 0.5), (0.01, 0.35), (0.08, 0.65))` (pairs (e, q)),
   the best converged start kept;
2. `sample_binomial`: per-sample logit-error and logit-balance effects with binomial kernels, pooled
   by a diagonal empirical-Bayes (normal measurement-error) approximation: each sample's
   uncertainty comes from the observed Hessian of its own likelihood with its genotype mixture
   profiled out, and the hyperparameters from the training data only;
3. `sample_joint`: those effects refitted jointly with the shared heterozygote concentration
   (starting at `EvidenceConfig.read_model.initial_heterozygote_concentration = 40.0`), keeping the
   empirical-Bayes prior.

A sample without training reads takes the pooled prior mean for its error and balance effects;
its genotype mixture stays uniform. When the pooled or the joint fit fails with an error, neither
enters. Stopping at an optimiser's iteration cap is not a failure: the stop reason is recorded and
never decides which fit is used. This is an approximation, not exact
Bayesian inference, and it does not guarantee identifiability at low depth.

### Folds, selection and fallback

Every `EvidenceConfig.read_model.marker_stride = 8`-th site (of the sites the filter keeps)
contributes per-sample histograms of (depth, ALT) cells with depth 1 to
`EvidenceConfig.read_model.maximum_fit_depth = 48`; deeper cells are left out of fitting only. The
sites fall into four physical folds, floor(POS / `EvidenceConfig.read_model.fold_window_bp = 1000000`)
mod 4. Folds 0 and 1 fit every candidate; fold 2 selects among the candidates with a finite
predictive log likelihood (each scored with its training genotype mixtures): candidates within one
nat of the best (`evidence.read_model.calibrate._SELECTION_TIE_NATS = 1.0`) are tied and the
simplest of them is applied, so a more complex model must predict the selection fold better by
more than a likelihood ratio of e; fold 3 is an untouched diagnostic
that never chooses parameters or the model. Every sample is calibrated, flagged samples included.

The selected training fit is applied to every cell of the contig without a full-data refit, so
some of the reads the likelihoods describe also trained the model: this is not out-of-fold
application. The fixed binomial model, e = `EvidenceConfig.fixed_error = 0.02` and q = 1/2 for
every sample, applies instead when calibration is disabled, when any fold lacks calibration
observations (short or sparse contigs), when no shared EM start converges, or when no candidate
has a finite selection score; the reason is recorded. No truth, pedigree or founder panel
enters estimation.

## Genotype likelihoods and the observation mask

Every cell of the contig, deep cells included, gets P(reads | g) of the applied model, normalised
over the three genotypes; no genotype prior is applied. Cells up to depth
`EvidenceConfig.read_model.lookup_depth = 64` come from an exact per-sample (depth, ALT) lookup
table and deeper cells are evaluated directly (a performance split, not an approximation). A
zero-depth cell is exactly (1/3, 1/3, 1/3). The stage checks every cell once (finite, non-negative,
row sum 1 within 1e-12, zero depth exactly 1/3) and fails otherwise; consumers rely on this
invariant. The observation mask is depth > 0.

## Discovery sites and blocks

A site is a discovery site when its total depth over all N analysed samples (flagged samples
included) is at least max(`EvidenceConfig.discovery_sites.minimum_total_depth = 5`,
`EvidenceConfig.discovery_sites.depth_per_sample = 0.02` x N). Raw discovery and local selection
fit and release founder alleles only at discovery sites; the other sites stay on the site axis for
the later stages. The per-sample term is an operating threshold, not derived from a model, and it
grows with the cohort size.
<!-- owner question (T2 Q1): discovery-site rule max(5, 0.02 N) kept from base; a cohort-size-independent rule would be a class-B change -->

The site axis is partitioned into consecutive, non-overlapping blocks of
`EvidenceConfig.block_sites = 200` sites (the last one shorter), stored as [start, stop) site
indices. These are the units of [discovery](discovery.md) and
[local selection](local_selection.md).

## Implementation and cost

| step | module | parallel structure | cost |
| --- | --- | --- | --- |
| loading | `evidence/vcf.py` | worker processes read disjoint physical windows (a record belongs to the window containing its POS) | O(N L) |
| site filter | `evidence/site_quality.py` | per-site genotype EM parallel over sites; elementwise mixture terms in thread chunks | O(N L) (at most 60 EM iterations per site) plus O(L) per mixture iteration |
| sample QC | `evidence/sample_quality.py` | vectorised | O(N L), then per-sample fits over private cells |
| read model | `evidence/read_model/fit.py`, `evidence/read_model/calibrate.py` | per-sample fits in threads; objectives parallel over samples | histograms O(N L / 8); each objective O(N C) with C <= 1,224 (depth, ALT) cells, independent of L |
| likelihoods | `evidence/read_model/calibrate.py`, `evidence/read_model/emission.py` | lookup tables per sample; gather over sample x 4,096-site tiles | O(N L) |

The stage has no dependence on the founder count K. Memory is dominated by the counts (8 bytes per
cell) and the likelihoods (24 bytes per cell) of the contig; `founder_view()` copies the arrays only
when some sample is flagged. Each contig is one record (`evidence`); a contig recorded under the same
identity is not recomputed, and a different identity refuses to resume (use a new output root).
The driver may prepare the next contig on a helper thread while discovery runs the current one.
Results do not depend on the CPU budget: every reduction over samples or sites runs in a fixed
order, and the window partition of the loader does not change the records read.

Convention reversal (`HAPLOTYPES_TIE_REVERSE=1`, release 2.2): the run's canonical sample axis is reverse name order
(maths.ties.run_sample_order); the read model keeps the last instead of the first EM start whose converged log
likelihood lies in the band of the best; the site filter's histogram mode is the last instead of the first of
exactly tied maximal bins; the sample-QC minor allele at exactly equal ALT and REF read totals is REF instead of
ALT. Not reversed (stated preferences): the simplest read model among equally good candidates and the hyperfit
boundary at tau = 0.

## Interpretation and limitations

- The likelihoods are P(reads | genotype) of one fitted model per contig. Model selection is
  predictive, not a significance test, and a better marginal fit to the allele depths is not proof
  of better founder reconstruction.
- Homozygote kernels are binomial: reads that contradict a homozygous genotype more often than the
  sample's error rate allows (correlated read errors, mapping or structural artefacts at a site) are
  not modelled, so such cells can look heterozygous. Site-specific mapping biases, asymmetric
  homozygote errors, variant ascertainment and dependent errors are not modelled.
- The joint fit can be flat in kappa (most of all without heterozygote overdispersion) and in the
  error logits of low-depth samples, so its endpoint and selection score move slightly with rounding
  (up to about 0.3 nats on the validation data), which moves individual likelihoods slightly. The
  one-nat tie rule keeps such movements from deciding between candidates unless a gap lies within
  rounding of one nat.
- Optimiser bounds, starts, tolerances and the sampling stride are numerical choices, not
  biological estimates.
- The site filter's depth rules cannot distinguish benign sample-specific missingness from mapping
  artefacts, and r5 removes sites without segregating information by design.
- The contamination QC detects foreign DNA through private alleles only: contamination shared by
  several libraries from one source is seen in part, and DNA from another analysed sample brings
  no private alleles, so contamination within the cohort is not detected.
- Calibration cannot resolve ancestral phase that the descendants do not identify.
