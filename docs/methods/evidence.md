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

**Retained r4 limitation and deferred replacement.** The evidence-only memory refactor keeps
base23's r4 model, eligibility, fit and gate unchanged. The clean class is exactly binomial,
so it cannot represent intrinsic variation in clean site balance. Its separation scale uses
median pooled heterozygote depth H (not individual sample coverage) and shrinks as H increases.
At TroMau coverage the fitted components do not meet the three-binomial-SD gate, so r4 does not
apply. On chr1/chr3/chr8, median H is approximately 632/600/636 and the artefact widths are
2.18/2.56/2.26 binomial SDs. This inactivity does not establish an absence of allele-balance
artefacts. Increasing coverage or cohort size can activate the gate even for unchanged latent
allele-fraction distributions.

A proposed replacement fits clean and artefact beta-binomial components, including exact
binomial limits, and assesses separation using the clean component's intrinsic allele-fraction
SD. The bounded diagnosis supported this change through read-quality and painting-conditioned
consistency checks. It is deferred in this variant: integration exposed a pre-existing
family-phase limitation for siblings with indistinguishable discrete coordinate keys. That
limitation requires work beyond the evidence stage; it is not evidence that the r4 diagnosis
was wrong. The replacement is not used by this implementation.

The r1/r2 diagnosis also remains unchanged: on bounded TroMau contigs, sites removed by the
depth rules had worse read balance, homozygote error and painting-conditioned founder
consistency, including matched-depth comparisons and thinning controls. Broad relaxation failed
those checks, so the depth rules are retained. These are internal consistency checks, not
independent biological truth or proof of a specific mapping artefact. They did not train the
filter or select its thresholds.

**Bounded memory and parallel execution.** Depth calculations and sparse QC visit bounded
site tiles, retaining O(M) site summaries rather than full float64 sample-by-input-site
intermediates. Genotype classification reads integer counts directly, with sample vectors
private to each worker. Site tasks enter one shared queue under the run's live CPU budget,
largest spans first. Workers immediately take another task when they finish; the queue can
admit more workers when a chromosome lease grows. The task count bounds worker scratch.
The same queue recomputes r5 among unflagged samples. Fixed read likelihoods are evaluated
once per site, outside the frequency EM iterations, preserving sample order and convergence.

The expensive sitewise r3/r4 mixture likelihoods run in compiled parallel kernels calling
SciPy's scalar special functions. Each objective evaluation reclaims the currently available
CPU allowance, with no fixed division between stages or fits. r4 remains dependent on the
completed r3 classification. Parameter updates and inexpensive reductions retain the original
order and optimizer; the scientific model and thresholds are unchanged. The compiled
negative-binomial arithmetic can differ from NumPy by rounding, which is checked against
SciPy and complete filter records in the bounded validation.

`peak_scratch_bytes(n_samples, n_input_sites)` supplies the complete conservative workspace
reservation for site vectors, bounded depth tiles, per-worker genotype scratch and sparse
sample QC. It excludes caller-owned counts and later evidence arrays. No adaptive allocation
or extra reservation is needed: at N=10,000 and M=749,986, the bound remains 539,010,048 bytes.
The integrated evidence stage reserves this exported bound before calling the filter.

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
is applied. Collection visits bounded tiles and stores only the at-most-two carriers at each
near-private site, including any partner's counts. Private-cell/event fitting, the robust-z
rule and the r4 balance supplied to QC are unchanged in this variant; bounded-contig QC records
match fresh base23 execution exactly.

**Inherited segments (reported, never flagged).** Near-private cells (the sample and at most one
partner carry the allele) whose reads favour the heterozygous fraction over h/2 are grouped per
(sample, partner) into fragment events and into regions of events less than
`EvidenceConfig.sample_quality.segment_gap_bp = 1000000` bp apart. Regions with at least
`EvidenceConfig.sample_quality.segment_minimum_events = 3.0` events are listed for unflagged samples
with the region's near-private cell count and the sample's and partner's minor-read fractions.
Such segments, shared by a pair of samples or private to one, can explain added founder rows and
pedigree misfits; they are not evidence of contamination.

**Effect of a flag.** On the contig where it is flagged, a sample leaves founder construction:
`ContigEvidence.founder_view()` treats it as effectively missing observations. The integrated
memory implementation shares immutable counts and likelihoods, copies the observation mask,
and marks excluded samples unobserved; consumers materialize neutral values where needed.
Discovery, local selection, assembly and the founder refit read that view. The site axis, the discovery sites and the blocks are unchanged (they
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
   each successful start finished at stationarity under the same training likelihood and the
   existing Newton convergence, curvature, bounds and backtracking policy. The two shared
   parameters and each sample's nuisance mixture are optimized jointly. Numerical endpoint
   equivalence requires pairwise relative shared-logit distances at most1e-13 and nuisance-logit
   distances at most1e-8, using max(1,abs(x),abs(y)) as the coordinate scale. These reuse the
   solver's small-step and stagnation resolutions; the latter is an equivalence rule, not an
   endpoint-error bound. A completed EM endpoint enters only if its polish converges;
   EM and polish failures are recorded as failed starts. Among equivalent endpoints, maximum
   training log likelihood wins; exact ties use the lexicographically smallest full parameter
   vector. Training histogram rows are evaluated in content order and nuisance rows restored
   afterward, making reductions and the parameter-vector representation independent of sample
   order. Held-out data and sample names do not determine this order. Distinct endpoints or
   failed polishing follow the recorded per-contig policy below;
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

A failed shared-binomial polish counts as a failed EM start. Among the remaining starts,
maximum training log likelihood wins, with the same lexicographic tie rule within equivalent
endpoints. If distinct endpoints remain, that winner is used only if its training log likelihood
exceeds every distinct competitor by more than one nat (the existing read-model tie band).
Otherwise the entire contig uses the fixed binomial model with reason
`shared_endpoints_unresolved`. When no start survives, the existing no-converged-shared-start
fallback applies. The sample-binomial fit initializes from the shared fit, and the joint fit from the
sample-binomial fit; neither is fitted when the shared fit is unresolved. Selection and diagnostic
folds do not resolve competing training endpoints.

These failures do not stop the run. Failed starts and decisions among distinct endpoints produce
a log warning, a calibration warning in the run summary, and evidence-report diagnostics:
initial starts, available endpoint vectors in training-histogram content order, training log
likelihoods, polish termination reasons, pairwise relative distances between successful endpoints,
and the decision reason. Successful-start indices map those distances back to the start records.
Normal contigs keep the same fit, likelihoods and report values.

## Genotype likelihoods and the observation mask

Every cell of the contig, deep cells included, gets P(reads | g) of the applied model, normalised
over the three genotypes; no genotype prior is applied. Cells up to depth
`EvidenceConfig.read_model.lookup_depth = 64` come from an exact per-sample (depth, ALT) lookup
table and deeper cells are evaluated directly (a performance split, not an approximation). A
zero-depth cell is exactly (1/3, 1/3, 1/3). The stage checks every cell once (finite, non-negative,
row sum 1 within 1e-12, zero depth exactly 1/3) and fails otherwise; consumers rely on this
invariant. The observation mask is depth > 0.

## Discovery sites and blocks

A site is a discovery site when its total depth over all analysed samples (flagged samples
included) is at least `EvidenceConfig.discovery_sites.minimum_total_depth = 5`. This fixed
read-support floor does not grow with cohort size: adding samples without reads at a site cannot
make that site ineligible. Five reads is an operating minimum, not a claim of allele certainty;
the genotype likelihoods retain the uncertainty and downstream release rules decide allele support.
Raw discovery and local selection fit and release founder alleles only at discovery sites; other
sites stay on the site axis for later filling and inference. The rule changes neither the
site-quality filter nor block boundaries.

The site axis is partitioned into consecutive, non-overlapping blocks of
`EvidenceConfig.block_sites = 200` sites (the last one shorter), stored as [start, stop) site
indices. These are the units of [discovery](discovery.md) and
[local selection](local_selection.md).

## Implementation and cost

| step | module | parallel structure | cost |
| --- | --- | --- | --- |
| loading | `evidence/vcf.py` | worker processes scan disjoint physical windows, then decode directly into one shared sample-major array | O(N M_in); two BCF passes, no count-array result transfers |
| site filter | `evidence/site_quality.py` | per-site genotype EM parallel over sites; elementwise mixture terms in thread chunks | O(N M_in) (at most 60 EM iterations per site) plus O(M_in) per mixture iteration |
| sample QC | `evidence/sample_quality.py` | vectorised | O(N M_in), then per-sample fits over private cells |
| read model | `evidence/read_model/fit.py`, `evidence/read_model/calibrate.py` | per-sample fits in threads; objectives parallel over samples | histograms O(N M_kept / 8); each objective O(N C), C <= 1,224, independent of chromosome length |
| likelihoods | `evidence/read_model/emission.py` | samples in parallel; a lookup/prefix table per active sample | O(N M_kept) |

The stage has no dependence on founder count K. Here N is sample count, M_in the input SNP count,
and M_kept the retained site count. Retained counts, likelihoods and observations occupy respectively
8, 24 and 1 byte per retained cell, plus site/sample vectors and the report. The loader writes final
C-order int32 counts directly; it retains O(M_in) metadata and one record's depths per worker instead
of whole-window results and concatenation/filter/transpose copies. Canonical sample ordering still
requires a second count array while permuting, so source loading can peak at 16 N M_in bytes.

Site selection writes one C-order count array. The driver releases input counts immediately after
selection, before calibration/emission. Read-model histogram construction retains the stride-site
int64 counts/depths and four square histograms, then a compact histogram copy: a conservative bound
beyond kept counts is 27 N ceil(M_kept/8) + 32 N (49^2 + C) bytes. Fitting remains O(N C).
Likelihood emission allocates its 24 N M_kept output once. Lookup storage is now
24 T (cap+1)^2 bytes for T active samples, instead of 24 N (cap+1)^2; prefix tables use at most
128 T (D+1) bytes for deepest cell D. Prefix sums and cell normalization retain their original
operation order. The observation-mask kernel allocates one bool matrix instead of three.

When a sample is flagged, `founder_view()` shares the immutable counts and likelihoods and copies
only the observation mask (1 byte per retained cell, against 33 for a full copy with counts or 25 without).
This represents excluded rows as missing: consumers must consult the mask before using raw values.
Discovery and assembly materialize neutral likelihoods where needed; founder refit skips unobserved
cells. Local selection neutralizes its existing private block copy before computing its persisted
search key, preserving both numerical inputs and cache/reuse decisions. The unchanged recorded
evidence still includes every sample for calibration, painting and later stages.

Each contig is one `evidence` record. The existing codec streams bounded chunks rather than copying
a full serialized contig. Output counts, site selection, preparation phases, the founder mask and
record compression scratch use governor reservations. Reservations describe planned allocations,
not fixed fractions of free memory. `preparation_bytes` uses the filter's `peak_scratch_bytes(N, M_in)` estimate when exported, with the
measured base23 filter/QC estimate as fallback; it does not describe retained evidence size.

The driver prepares at most one next contig on a helper thread while discovering the current one.
For flagged equal-size contigs with counts, current evidence plus its founder view drops from
66 N M_kept to 34 N M_kept bytes. A ready next contig adds 33 N M_kept; while being prepared, its
transient peak and discovery's worker reservations must also fit. At N=10,000 and M_kept=127,172,
the current evidence/view pair is 43.24 GB instead of 83.93 GB (decimal). At three times as many
retained sites it is 129.72 GB instead of 251.80 GB. These savings do not eliminate the site filter's
separate input-site workspace.

Bounded base23 comparisons on TroMau chr3/chr19 (one flagged sample each), std_n80 chr1/chr2, and
N=10,000 synthetic evidence preserve every evidence array and report value apart from timing.
The generated N=10,000, M_in=2,048 BCF reduced decoder peak parent memory above initialization
from 32.66 to 8.25 bytes/input cell. Six-worker chr19 decode wall time was unchanged; a three-worker
case traded about 0.2–0.3 seconds for the smaller memory footprint. A bounded three-worker decode
plus three-worker discovery test did not reproduce the production 3–4x decode slowdown. Parent
decode CPU fell about 98%, but the metadata scan adds worker CPU. The driver still gives discovery
and evidence independent full budgets; production-scale contention needs the coordinator's run.

A record under the same identity resumes; a different identity refuses to resume (use a new output
root). Results do not depend on CPU budget: reductions over samples/sites retain fixed order and
every record belongs to the physical window containing its POS.

Convention reversal (`HAPLOTYPES_TIE_REVERSE=1`, release 2.2): the run's canonical sample axis is reverse name order
(maths.ties.run_sample_order); shared-binomial read-model starts must agree after stationary polishing,
so reversal does not select an EM endpoint; the site filter's histogram mode is the last instead of the first of
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
  error logits of low-depth samples. The release 2.1 Newton polish finishes the L-BFGS-B fits at
  stationarity under the existing bounds and stopping rules; the pre-polish 0.3-nat score movement
  is not a bound on the current implementation. Retained environment-comparison records across
  all 22 TroMau contigs show a maximum relative applied-GL change of 3.67749e-10 after polishing
  (chr22), with every selected model unchanged. Flat directions can still amplify rounding, so
  neither numerical uniqueness nor invariance of every downstream decision follows from this
  comparison. The one-nat selection rule is unchanged.
- Optimiser bounds, starts, tolerances and the sampling stride are numerical choices, not
  biological estimates.
- The site filter's depth rules cannot distinguish benign sample-specific missingness from mapping
  artefacts, and r5 removes sites without segregating information by design.
- The contamination QC detects foreign DNA through private alleles only: contamination shared by
  several libraries from one source is seen in part, and DNA from another analysed sample brings
  no private alleles, so contamination within the cohort is not detected.
- Calibration cannot resolve ancestral phase that the descendants do not identify.

### Shared-binomial stationarity validation

The standard chr2 discovery reproducer exposes why a likelihood band alone is insufficient:
finite EM endpoints inside that band produced different calibrated likelihoods and K5/K6
founder selections under reversal. All three starts and both sample orders converge to a
common stationary optimum within the existing relative tolerance of 1e-13 after three Newton
steps. The largest parameter distance is 1.1e-14; the largest applied likelihood change from
2.2 on the captured block is 4.38e-5. This is a numerical optimization correction under the
same training likelihood, not a mixture of read models. The training/held-out folds and model
selection rule are unchanged. Whole-dataset parameter, likelihood, truth and stability gates
remain separate from this bounded reproducer.

The shared Hessian has two common coordinates and a two-coordinate nuisance block per sample.
Central differences of analytic gradient rows construct it in eight gradient passes. A Schur
solve avoids a dense matrix whose dimension grows with sample count; independent dense-Newton
fixtures check endpoint agreement, including an empty sample. The existing Newton stopping and
backtracking rules are reused. Per-start objectives, iterations, convergence and endpoint
distances and small-step/stagnation termination reasons are recorded. The endpoint equivalence
rule does not alter Newton's updates, stopping conditions, score bands, priors or held-out weights.

The 110-contig census exposed seven cases whose nuisance logits differ by up to 2.12e-11 despite
successful polishing; shared logits differ at most 5.61e-15 and applied likelihoods at most 1.50e-14.
Eighteen of their 21 starts stop by stagnation. These observations motivate the explicit equivalence
rule; they do not prove a unique optimum. Maximum training score and the exact lexicographic tie
rule remove start-order dependence within that family. A separate content order for training
histograms removes last-bit sample-reduction differences. Full-model and downstream controls
remain necessary because nuisance mixtures also enter model selection and later fits.

The final census passes the approved equivalence test on all 110 contigs: all 330 original
EM starts and their polishes succeed, with no failed or skipped starts. Eight critical
contigs, including the seven original failures and the standard chr2 reproducer, were
calibrated under 11 profiles each. Start permutations, duplicate starts, convention reversal,
sample reversal and sample permutation give bitwise-identical aligned GLs in all 88 profiles.
This checks deterministic representation within the tested endpoint families; it does not
prove mathematical uniqueness or downstream accuracy.

Compared with 2.2, full recalibration preserves the selected model on ten of eleven datasets.
K20/K30 retain their fixed fallback and exactly unchanged GLs. Standard and stress maximum
applied GL changes are 6.33e-5 and 3.57e-5. In sim_s10001, chr1 changes from shared binomial
to sample binomial, while chr6 and chr11 change in the opposite direction; the largest GL
change is 0.00679817. Shared training optimization changes held-out scores enough to cross
the unchanged one-nat model-selection rule. Full regenerated-evidence truth gates are required
for these discrete model changes. No held-out score enters the within-shared endpoint choice.

### Shared-binomial Newton curvature and bounds (2.3)

The nuisance-mixture Hessian is evaluated analytically as the histogram-weighted difference of prior and posterior categorical covariance matrices. Pair-covariance sums retain nearly absent genotype categories; the shared/nuisance coupling is the analytic posterior score covariance. The shared two-parameter block retains central finite differences.

The Newton direction minimizes its quadratic over the existing parameter bounds using a structured active-set Schur solve. All simultaneous blockers enter together; multiplier ties are handled as complete classes. A nonpositive reduced Hessian or a repeated active face is an explicit fitting failure. This corrects the N-series near-boundary failure caused by separately clipping an unconstrained joint direction.

The training likelihood, parameter bounds, backtracking limit, Newton convergence/iteration limits, EM starts, endpoint-equivalence resolutions, content representative, held-out model selection and applied evidence functional are unchanged. Release 2.4 adds the per-contig failure policy described above. No damping, new tolerance, prior, or rescue start is introduced. Each reduced-face solve is linear in sample count at fixed local dimension; iteration and active-face counts are reported as solver behavior rather than a universal scaling proof.

The coordinator accepted the measured shared-curvature implementation change of at most
1.97e-11 in applied GLs (TroMau chr22) for release 2.4. All selected models and measured
calibration decisions were unchanged. The matched-host comparison traces the change from
shared nuisance logits through profile-Hessian uncertainty and the pooled balance hypervariance;
the shared error and balance logits are bitwise identical. The maximum absolute GL change is
accepted in the context of the existing 1e-9 relative GL-noise validation cell. The 1.97e-11
number is an absolute change; the relative perturbation cell is a separate stability experiment.
This acceptance introduces no new tolerance and does not infer downstream identity from small
GL changes alone.
