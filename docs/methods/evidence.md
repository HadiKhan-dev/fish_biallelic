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

The loader reads POS, REF, ALT and FORMAT/AD only. The input's FILTER and QUAL are ignored: a record
that an upstream filter marked as failing is analysed like any other. To keep only records that
passed an upstream filter, pre-filter the file (`bcftools view -f PASS`). The FILTER written by the
export is this package's own site filter (below), not the input's.

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
| r4 | ALT fraction in samples classified heterozygous | of two beta-binomial classes, the minority artefact class explains it better than the majority clean class |
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

- r3, among r1/r2-passing sites with N_l >= 1: clean W ~ Poisson(lambda_l + kappa_l eps0), artefact
  W ~ NegBin(mean lambda_l + kappa_l eps1, shape k, clipped to [1e-3, 1e4]). Release 2.18: W weights each
  cell's reads by the classification's posteriors, so a heterozygote whose reads show its minor allele once,
  frequent at low depth, also adds to W. Under a clean site with error rate e the expectation of W, given the site's
  fitted genotype frequencies F_l, is to first order in e lambda_l + kappa_l e, with w(a, d) = p0 a + p2 (d - a) a cell's posterior-weighted wrong reads
  (h = the classification's heterozygote balance): lambda_l = sum over the site's cells of
  F_l1 sum_a Bin(a; d, h) w(a, d) + F_l0 w(0, d) + F_l2 w(d, d), the heterozygote reads the classification counts
  as homozygote errors and, at depth 1-2, the reads of error-free homozygotes weighted by the other homozygote's
  posterior; kappa_l = sum over cells of d (F_l0 (w(1, d) - w(0, d)) + F_l2 (w(d - 1, d) - w(d, d))), the change one
  error read makes. A site whose cells carry no first-order information on e (kappa_l below 10^-9 of its
  homozygote depth N_l, for example only depth-1 cells at a balanced site) is not scored by r3; the floor also
  excludes some small positive slopes, whose whole mean change over e in [0, 1/2] is below 5·10^-10 N_l. W is a
  posterior-weighted, non-integer count, so its Poisson and negative-binomial likelihoods are approximations.
  The clean rate e0 is searched in [10^-12, 1/2]: a bound is returned when the concave objective's gradient keeps
  its sign there, otherwise Newton steps that keep a bracket of the root, with bisection when a step leaves it.
  Until 2.17 the
  clean mean was N_l eps0 (lambda = 0, kappa = N), which left the leaked reads in the fitted rate: at 4-6x the
  leakage depends on each site's genotype frequencies, the mixture separated sites by frequency rather than by
  quality, and r3 removed 79-99% of the sites of a clean simulation (2.18's r3 removes none; at 7x and above
  neither r3 removed any). On real TroMau (about 13x) the leaked reads are 22-23% of W, the clean rate falls from about
  0.0014 to 0.0003 and r3 changes on about 2% of the sites it tests: the sites it now removes that 2.17 kept fit
  the released painting worse than kept sites (0.55-0.59 against 0.08 nats per sample);
- r4, among r1-r3-passing sites with H_l >= 1 (release 3.4): A_l ~ Binomial(H_l, theta_l), with the
  site's latent heterozygote ALT fraction theta_l drawn from a mixture of two beta distributions, each
  with mean m and inverse concentration t (latent variance m (1 - m) t / (1 + t); t = 0 is exactly the
  binomial, so the clean sites' balances may vary or not). The clean class is the component with the
  larger weight (at exactly equal weights, the one with the smaller latent variance); its mean h0 is
  also the balance the sample QC uses. A_l and H_l are non-integer, so the binomial is written with
  gamma functions.

r3 is fitted by EM: closed-form mixing updates, the clean rate by the bracketed Newton search above
(closed form when no site has leakage), and an artefact M-step that maximises the weighted artefact
log likelihood over its two parameters (log rate and log shape) jointly within the box of the shape
clip, by projected Newton steps with analytic gradients and Hessians. An M-step stops when the full Newton step moves no transformed
parameter q by more than 1e-8 max(1, |q|); it is certified stationary when its projected gradient,
divided by the artefact class's posterior weight, is then at most 1e-9 (a dispersion held at its
clip needs only an outward gradient), and otherwise recorded as a small-step stop whose gradient
certificate is unmet (`precision`). Each mixture records its M-step outcomes (`m_steps`:
stationary, precision, line-search and iteration-cap counts, and the largest gradient over all
its M-steps). On TroMau in release 2.11, 2,060 of 3,236 M-steps (r3's and those of the then r4) were certified and the rest were small-step stops
(largest normalized gradient 3.1e-7); none failed its line search or reached the cap. The EM stops when the summed parameter change
falls below 1e-5, or after 300 iterations. Where the data hold no artefact
class (clean reads) the two components nearly coincide and the EM can run to its cap; each
iteration then costs a few likelihood passes. Releases 2.10 and earlier searched the unbounded
parameters with Nelder-Mead and then clipped the dispersion alone, which could leave the stored
rate or mean non-stationary for the stored dispersion and miss the maximum by more than the EM
tolerance, at about 100 likelihood passes per iteration. On TroMau the change moves the r3 flag
of one site on each of five contigs, each with a clean posterior of about 1/2 (0.49995 to
0.50002 in the comparisons); three of them become kept sites, each of which can also enter the
calibration sample (below).

r4 is fitted by maximum likelihood in all five parameters jointly (logit weight, the two logit means
and log1p(t / 0.001) for each component), within +/-18 on the logits and t <= 1e6, from five fixed
starts about the pooled mean: each by L-BFGS-B, restarted up to three times while its projected
gradient exceeds the certificate, then up to eight positive-curvature Newton steps on the score
(near the maximum, likelihood gains fall below floating-point resolution, so the score is solved
instead). The component likelihoods are written relative to Binomial(H, 1/2) with gamma-ratio
remainders (a Stirling series at small t), so non-integer counts and concentrations up to the
binomial limit keep full precision, and sites enter in canonical (H, A) order, so the fit does not
depend on site order or thread count. A start is certified when the projected gradient of the mean
per-site log likelihood is at most 1e-8; the certified start with the highest likelihood is used
(exact ties by its parameters, then its start). This certifies numerical stationarity, not a global
maximum or biological class labels. If no start is certified, r4 removes nothing and its record
says why; the best available fit still supplies h0 to sample QC. The record lists every start.
An exact t = 0 is a valid binomial boundary. A positive component weight at its numerical bound
does not prevent a minority posterior above 0.5 when its likelihood ratio exceeds the prior odds;
a stationary endpoint also need not inherit the global maximum's nesting guarantee.

A site is flagged by r3 or r4 when its posterior probability of the clean class is below
`EvidenceConfig.site_quality.clean_posterior_level = 0.5`, and only when the artefact class
separates from the clean core (`EvidenceConfig.site_quality.minimum_artefact_separation = 3.0`):
eps1 >= 3 eps0 for r3; for r4 an artefact latent standard deviation, or a mean offset |m - h0|, of
at least 3 s, with s = max(the clean latent standard deviation, the binomial standard deviation
sqrt(h0 (1 - h0) / H) at the median heterozygote depth H of r4's sites). This is a conservative
contig-level working gate defined at typical pooled depth; it can miss well-resolved high-H sites.
It is not a per-site resolution limit or a guarantee on clean-site error: overlapping classes can
lose clean sites when the gate opens. With a binomial clean class, s is the binomial standard
deviation used by releases 2.18-3.3. A mixture whose artefact class does not separate removes nothing.

r4 also abstains when abs(h0 - 0.5) - abs(m - 0.5) >= 3 s: the majority mean is substantially farther
from 1/2 than the minority mean, contradicting the working clean label. The record reports
unresolved class labelling; the components are not relabelled. The fitted h0 still feeds sample QC,
now on all r1-r3-passing sites. This protects balanced minority sites, at the cost of abstaining on
some genuinely shifted clean cores; it does not identify which class is biologically clean.

**r4 before release 3.4.** Releases up to 3.3 modelled the clean class as exactly binomial, beside
one beta-binomial artefact class, so clean sites whose balance varies slightly had no place in it.
In a planted test (truth haplotypes of the TroMau-like simulation, depth 13, clean balance standard
deviation 0.02, 3% of sites with heterozygote ALT fraction 0.30) the gate stayed shut at N = 115,
and the biased sites were caught only by r3 (62-65%); at N = 2,500 the beta-binomial class absorbed
the clean bulk, the gate opened as the binomial standard deviation shrank with depth, r4 removed 94%
of the clean sites, and no site was left. A replacement prepared in 2.4 fitted two beta-binomials
but called the narrower component clean, and a tight bias class is the narrower one: it removed
94-98% of the clean sites at every N. Release 3.4 keeps that fit and identifies the clean class by
weight. In the same planted test (36 cells: N = 115, 580 and 2,500; clean balance standard deviation
0 or 0.02; a tight bias class, or a broad one of standard deviation 0.08; harmless per-cell
heterozygote overdispersion of standard deviation 0.05 or 0.10; two draws each) r4 flags at most
0.12% of the clean sites, those in one cell whose broad bias class reaches into the clean range, and
at most 0.05% elsewhere; no cell loses its kept sites to r4. With clean balances varying it catches
the tight bias class at N = 115 (with r1-r3, 96-97% against 62-65%) and the broad one (85-87%
against 56%). These clean-loss figures describe this planted matrix, not all class prevalences
or shapes. The stricter predeclared 0.1% clean-loss and no-recall-regression criteria were narrowly
exceeded: some boundary sites trade recall against clean-site retention.

On TroMau (116 samples, about 13x) the clean component holds 86-95% of r4's sites, with a
latent standard deviation of 0.002-0.007, and the artefact component (mean 0.47, latent standard
deviation 0.040-0.053) lies only 2.0-2.6 binomial standard deviations out in spread and 1.1-1.4 in
mean, so r4 removes nothing there. Scaled by the clean latent standard deviation alone,
as the 2.4 replacement proposed, the same fit would remove 1.7-5.5% of the kept sites. This
inactivity does not show that TroMau has no allele-balance artefacts: as a cohort grows, the
binomial standard deviation shrinks, and a fixed artefact spread separates sooner.

The E1-scale sensitivity arm would remove 2.69% overall, with a 3.08-fold larger painting deficit
under a fixed 0.5-balance read model. That is read/model inconsistency, compatible with completely
correct founder alleles and tracks: a known-truth fixture changing only heterozygote balance from
0.50 to 0.38 produced a 3.23-fold deficit while recovering every founder vector correctly.
Selection by extreme balance makes this same-read diagnostic partly self-selected. It does not
prove wrong founder alleles or mismapping, and does not justify adopting E1's scale on its own.

TroMau's measured cell-level latent SD of 0.022-0.033 is the fitted bulk dispersion among retained
r1-r4 sites and unflagged samples, under a truncated beta-binomial model. The likelihood normalizes
the read window (2 <= ALT <= depth - 2, depths 8-60), but not the preceding site and sample
selection. Site-level spread contributes to the estimate; selection and a single Beta fit to a
mixture or heavy tail can instead underestimate latent variance. It is conditional evidence of a
small bulk effect on present TroMau, not a distribution-free upper bound, clean-site accuracy
measurement or N = 2,500 safety certificate. The reported binomial likelihood contrast holds h at
the alternative estimate; it is not a maximized likelihood-ratio statistic without refitting h
under the null.

In planted Beta cell-noise tests, r3 loses clean sites at N = 2,500 as cell SD rises from 0.05 to
0.07, reaching 87% loss at 0.10. r3 is unchanged here. Extrapolation to larger real cohorts needs
unfiltered/filtered tails and depth/library strata. A future r3 repair needs per-cell dispersion,
not r4's pooled-site SD: independent deviations average down approximately as 1/n in variance
across n equally covered heterozygotes. TroMau has no truth; these observations establish
consistency and model-conditional evidence only.

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

The expensive sitewise r3 mixture likelihoods run in compiled parallel kernels calling
SciPy's scalar special functions. Each objective evaluation reclaims the currently available
CPU allowance, with no fixed division between stages or fits. r3's parameter updates and
inexpensive reductions retain the original order and optimizer. The compiled negative-binomial
arithmetic can differ from NumPy by rounding, which is checked against SciPy and complete filter
records in the bounded validation. r4's kernels are compiled and parallel over sites, take the
allowance available when its fit starts, and reduce per-site terms in a fixed order. r4 remains
dependent on the completed r3 classification.

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

Inside the site filter, each contig checks every analysed sample for a foreign-like minor-read component, the
signature of library contamination, and excludes a flagged sample from founder construction on that contig
(`evidence/sample_quality.py`). Foreign DNA at read fraction alpha < 1/2 shows, at a site where the foreign genome
carries an allele the library's own genome lacks, a minor-read fraction a = alpha (foreign homozygote) or alpha/2
(foreign heterozygote), at the same a wherever such sites fall. A library's own heterozygous allele that no other
library carries (an inherited segment absent from the other sequenced samples) shows its heterozygote balance instead,
and a local artefact (a collapsed repeat, a mismapped paralog, a private tract with mapping bias) whatever fraction it
has, in a few places. Only the first is flagged.

**Eligible cells.** At the SNPs that pass r1-r4 (whole analysed cohort), the cohort-minor allele is ALT when the
cohort's ALT reads do not exceed its REF reads. A library carries it at a site when its k minor reads of depth d are
improbable as errors at the scale of the cohort: P(K >= k | d, e) <= q / (N - 1), K ~ Binomial(d, e), with e r3's
clean rate eps0 (the classification error when r3 has no fit) and
q = `EvidenceConfig.sample_quality.carrier_tail_level = 0.05`. Under a homogeneous binomial error null, a site whose
allele nobody carries then shows a carrier anywhere with probability at most q, whatever N (union bound); a single
error read does not make a carrier, so private cells persist as the cohort grows. A cell (library, site) is
eligible when the library is observed there and no other library carries the minor allele: its minor reads can only
be errors, the library's own private heterozygous allele, or foreign reads. Every eligible cell enters, k = 0
included, so depth enters exactly.

**Composite weights.** Cells of a library closer than one sequenced fragment share reads. Each eligible cell weighs
1 / (the library's eligible cells in the same `EvidenceConfig.sample_quality.fragment_bp = 1000` bp window, counted
from the contig's first passing site, so translating every coordinate changes nothing), so a window counts once
however dense the SNPs: evidence and exposure are in the same units, and a local cluster of artefact cells is one
event.

**Cell model.** Let a_min = `EvidenceConfig.sample_quality.minimum_foreign_fraction = 0.01` / 2, a working
foreign-allele amplitude. Per library, an eligible cell in depth class j (the library's weighted eligible-cell depth
quantiles, `EvidenceConfig.sample_quality.heterozygote_depth_classes = 4`) has minor reads k with probability

    x0(k) = sum_u pi_ju Bin(k; d, eps_u) + sum_v w_jv BB(k; d, h_v^o, rho_v)       (no foreign component)
    x1(k) = the same + c Bin(k; d, a),   a in (a_min, a_hi]                           (foreign component)

with each class's weights (plus c under H1) summing to 1. The terms are:

- the library's own errors: a free mixture over a log grid of rates eps_u from e / 10 to a_min (neighbouring rates
  at most 1.6x apart), fitted per depth class. Any error rate of the library, or spread of error rates over its
  sites, up to a_min belongs to the null, so library-specific error is not read as foreign reads;
- the library's own private heterozygotes: beta-binomial components with ALT mean h_v and latent dispersion rho_v,
  rho_v in {0, 1e-4, 4e-4, 1.6e-3, 6.4e-3, 0.0256, 0.04} (latent SD up to 0.1 at h = 1/2). The means are r4's clean
  balance h0; when r4 records an unresolved class labelling or no certified fit, h0 is only one of two candidate clean
  balances, and the means are the ends and middle of the interval spanning r4's two component means (the QC records
  it and r4's reason). A private REF allele uses 1 - h_v with the same rho_v, so both tails are modelled alike in both
  orientations: complementing every count of a library leaves its likelihood unchanged. The weights may differ
  between depth classes, because mapping bias lowers a site's depth together with its allele fraction;
- c, the foreign-component share within the non-artefact mixture, common to depth classes and regions. Its
  marginal weight in region R is (1 - pi_R)c, which can vary by region;
- a, the foreign allele's read fraction, on G = 24 log-spaced values over (a_min, a_hi], with
  a_hi = f h_lo / 2 (f = `EvidenceConfig.sample_quality.heterozygote_floor_ratio = 0.5`; about 0.12). Grosser
  contamination is detected at the ceiling, its fraction reported as at least the ceiling.

**Local artefacts.** The contig's passing sites are cut into `EvidenceConfig.sample_quality.artefact_regions = 16`
regions of equal site counts (at fragment-window boundaries). In region R a cell is, with probability pi_R, a local
artefact whose minor fraction is modelled as uniform on [0, 1], so that its k is uniform on 0..d:
A(k) = 1 / (d + 1). Each region's artefact share pi_R is free under both hypotheses:

    L(H) = prod_R max_{pi_R} prod_{b in R} [ (1 - pi_R) x_H(b) + pi_R A(b) ]^{n_b}

The fraction a and the non-artefact share c are common. The marginal foreign weight (1 - pi_R)c can differ
between regions; no region must independently support contamination. Empty or ineligible regions provide no
penalty. A local excess can be absorbed by the regional shares, but absorption is not guaranteed: it depends on
the distribution of eligible weighted information, not physical coverage or passing-site counts. The uniform
artefact law and 16 regions are working modelling choices. In the selected 4/8/16/32-region study, flags agreed,
but statistics changed (one planted 2% case: 51.0 to 48.9; largest other TroMau statistic: 15.2 to 4.2).

The fit maximises each likelihood over the mixture weights, c and the regions' shares by generalised EM: each E-step
sets every region's pi_R to its exact maximum (a concave one-dimensional problem) and weighs each cell by its
posterior of being an ordinary cell; each M-step takes damped Newton steps on the weighted mixture problem, which is
concave in the weights (each class's sum constraint eliminated through its largest weight). The objective never
decreases. The alternatives are warm-started across the grid of a, and the null is also polished from the best
alternative's weights, reducing the risk of false evidence from incomplete null optimisation. Conditional
concavity and polishing do not certify a global joint optimum.

**Statistic and working threshold.** T = max over the grid of 2 (log L(H1, a) - log L(H0)). For one a, c = 0 lies on
the boundary, and the regular asymptotic law of the statistic is (chi2_0 + chi2_1) / 2 (Self and Liang 1987).
Bonferroni over the G values of a and the union bound over the N libraries give the threshold: T >= u_N with
P(chi2_1 >= u_N) = 2 q / (G N), with q = `carrier_tail_level`. That is 17.1 at N 115, 20.1 at N 580 and 22.9 at
N 2,500. This level is nominal only for a regular, correctly specified likelihood of independent windows. The
composite weights, mixing weights on their boundary, the artefact components and misspecification all break it. u_N
is therefore a working threshold, supported by the clean false-flag rate measured on simulated libraries, not a
certified family-wise level.

**What a flag means.** A pooled likelihood preference for the foreign-component model above a_min over the
fitted error, own-heterozygote and artefact mixture. A flag excludes the library from founder construction on the
contig; it is not a contamination diagnosis. Elevated own error, biased private heterozygotes and dispersed
variable-fraction artefacts can also trigger exclusion. The reported `foreign_share` is c from the winning
alternative; reported artefact shares describe the null. They are not parts of one fitted decomposition.
The fitted fraction is not a calibrated contamination estimate: depth correlated with donor-private status can
push a much smaller true fraction to the fitted ceiling.

The amplitude a_min is a working effect threshold, not a derived safety boundary. The founder-construction and
painting emissions mix every observed site's genotype evidence with a uniform genotype error of mass r = 0.01, which
limits the influence of discordant observations. Where foreign DNA at alpha is homozygous for an allele the sample
lacks, a cell of depth d shows Bin(d, alpha) foreign reads. Two or more contradict a homozygote decisively at depths
of about 4-16x (at clean error rates 1e-4 to 1e-3), with probability about C(d, 2) alpha^2. That is of the order of
r at alpha 0.8-1.4% over 11-18x (alpha 1%: 0.72% at 13x; alpha 2%: 2.7%). A foreign allele shows at alpha or alpha/2,
so a_min = 0.5% aims at 1% contamination of either zygosity. A flag does not establish alpha >= 1%, and sensitivity
near the amplitude is measured, not guaranteed.

The QC is not applied, nothing is flagged and the reason is recorded, in any of these cases:
- fewer than `EvidenceConfig.sample_quality.minimum_samples = 20` samples have observed cells;
- there is no eligible cell;
- the cohort's clean error is not below a_min / 2 (errors and foreign alleles could not separate);
- the foreign range ends below 1.25 a_min.

No labels, pedigree, benchmark or cohort-relative scale enter. The decision is per contig, so a contig's result does
not depend on which other contigs a run includes. The QC runs only where the site filter is applied. Collection
visits bounded tiles of the passing sites and keeps per-library histograms of (orientation, depth, minor reads) with
summed weights per region, plus at most two near-private carriers per site; the profiles run per library in
parallel.

**Inherited segments (reported, never flagged).** Near-private carrier cells (the sample and at most one
partner carry the allele, carriers as above) whose reads favour the heterozygous fraction over h/2 are grouped per
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
Discovery, local selection, assembly, founder refinement and the founder refit read that view. The site axis, the discovery sites and the blocks are unchanged (they
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

Calibration uses every kept site. Each contributes per-sample histograms of (depth, ALT) cells with depth 1 to
`EvidenceConfig.read_model.maximum_fit_depth = 48`; deeper cells are left out of fitting only. The sites fall into
four physical folds, floor(POS / `EvidenceConfig.read_model.fold_window_bp = 1000000`) mod 4: folds 0 and 1 train,
folds 2 and 3 are held out. The histograms are built one sample at a time, so their memory does not grow with the
number of sites. Folds 0 and 1 fit every candidate.

Each more complex candidate in turn (the sample binomial, then the sample joint model) replaces the currently accepted
one only when it predicts the held-out folds better (each candidate scored with its training genotype mixtures) by at
least `EvidenceConfig.read_model.selection_z = 2.0` standard errors; a rejected candidate does not stop the next
comparison, which is then against the model still accepted. The standard error is not the per-site one. Sites are not
independent observations: nearby sites can share sequenced fragments, and in a cross the genotype composition of the
samples changes along the chromosome with their ancestry, so the per-site score differences are correlated over long
distances. On six TroMau contigs the standard error of a contig's total held-out gain (sample binomial against
shared), estimated from sums over windows of increasing length, grows 1.2- to 1.3-fold from single sites to 1-kb
windows (a scale at which reads may be shared) and 4- to 16-fold to 2-Mb windows, and on five of them further at 8 Mb:
windows do not show where the dependence ends, and they also mix in regional differences of density, depth and fit.
The error is therefore estimated from the gains of the B held-out stretches that hold admitted cells (folds 2 and 3 of
each four-window cycle, 2 Mb each by default) as sqrt(B var(stretch gains)); with fewer than
`EvidenceConfig.read_model.selection_minimum_blocks = 4` such stretches the configured rule does not use the estimate
and the shared binomial is kept (an explicit conservative fallback, not evidence that it fits better). This error
assumes nearly independent, comparable stretches and ignores the uncertainty of the training fits; the z rule is a
preference for the simpler model, not a significance test with a calibrated error rate (B is small, about nine per
TroMau contig). Each comparison records its stretches' admitted cells, the largest stretch's share, an
exposure-centred error and how many leave-one-stretch-out decisions differ, as sensitivity diagnostics.

Releases 2.3 to 2.11 fitted on one kept site in eight and selected by the summed held-out score of one fold with a
one-nat band. Both choices made the result depend on which sites were sampled: redrawing the sample changed per-sample
error estimates by about 10% (a contig's mean error logit by about 0.1) and the selected model on 7 of 22 TroMau
contigs. With all-site fits the joint model's held-out gain over the sample binomial is near zero on most TroMau
contigs (chr10 -0.3 +- 22, chr16 2 +- 6, chr22 9 +- 16 nats), well inside the noise of a one-nat band. Fitting on all
sites removes the sample; the stretch-based error keeps such near-zero gains on the simpler model (chr3, 688 +- 218,
keeps the joint model). Every sample is calibrated, flagged samples included.

The selected training fit is applied to every cell of the contig without a full-data refit, so
some of the reads the likelihoods describe also trained the model: this is not out-of-fold
application. The fixed binomial model, e = `EvidenceConfig.fixed_error = 0.02` and q = 1/2 for
every sample, applies instead when calibration is disabled, when the training or held-out folds lack
calibration observations (short or sparse contigs), when no shared EM start converges, or when the shared model
has no finite held-out score; the reason is recorded. No truth, pedigree or founder panel
enters estimation.

A failed shared-binomial polish counts as a failed EM start. Among the remaining starts,
maximum training log likelihood wins, with the same lexicographic tie rule within equivalent
endpoints. If distinct endpoints remain, that winner is used only if its training log likelihood
exceeds every distinct competitor by more than one nat (the existing read-model tie band).
Otherwise the entire contig uses the fixed binomial model with reason
`shared_endpoints_unresolved`. When no start survives, the existing no-converged-shared-start
fallback applies. The sample-binomial fit initializes from the shared fit, and the joint fit from the
sample-binomial fit; neither is fitted when the shared fit is unresolved. The held-out folds do not
resolve competing training endpoints.

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
| sample QC | `evidence/sample_quality.py` | vectorised collection; per-library profiles in parallel | O(N M_in), then per library O(G I (B Q^2 + (J Q)^3 + R_B S_b)) for I EM iterations (each a few Newton steps), B histogram bins, R_B region-bin pairs, S_b = 48 share bisections, G = 24 fractions, J <= 4 depth classes and Q = E + V components (E error rates, about 12; V = 7 or 21 own-heterozygote components); measured at N 2,500 on 16 cores: about 40 s for a contig of 150,000 sites |
| read model | `evidence/read_model/fit.py`, `evidence/read_model/calibrate.py` | per-sample fits in threads; objectives parallel over samples | histograms and held-out stretch scores O(N M_kept); each objective O(N C), C <= 1,224, independent of chromosome length |
| likelihoods | `evidence/read_model/emission.py` | samples in parallel; a lookup/prefix table per active sample | O(N M_kept) |

The stage has no dependence on founder count K. Here N is sample count, M_in the input SNP count,
and M_kept the retained site count. Retained counts, likelihoods and observations occupy respectively
8, 24 and 1 byte per retained cell, plus site/sample vectors and the report. The loader writes final
C-order int32 counts directly; it retains O(M_in) metadata and one record's depths per worker instead
of whole-window results and concatenation/filter/transpose copies. Canonical sample ordering still
requires a second count array while permuting, so source loading can peak at 16 N M_in bytes.

Site selection writes one C-order count array. The driver releases input counts immediately after
selection, before calibration/emission. Read-model histogram construction holds one sample's int64
depths and cell keys at a time, two square histograms and their compact copies, and, for the held-out
stretch scores, three per-cell predictive tables: a conservative bound beyond kept counts is
40 M_kept + 16 N (49^2 + C) + 24 N C bytes. Fitting remains O(N C).
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
  comparison.
- Optimiser bounds, starts and tolerances are numerical choices, not biological estimates; the
  selection threshold (two standard errors over the held-out stretches) is a preference for the
  simpler model applied to summed marginal predictive scores, not a confidence level; the stretch
  error is estimated from about nine stretches per contig and assumes the stretches are nearly
  independent. A contig whose gain lies near the threshold can still change its selected model under
  small input changes (such selections are counted in the ties report).
- The per-sample error and balance effects are empirical-Bayes estimates: each sample's logit is shrunk toward a
  fitted population mean with a weight set by its own sampling variance. Their values therefore depend on how much
  data the fit sees. On TroMau, fitting one kept site in eight (releases 2.3 to 2.11) gave a median per-sample error
  about three times the all-site fit's (chr16: 4.4e-5 against 1.4e-5), the same in every eighth of the sites. At a
  few tens of thousands of observed cells per sample, a sample whose error rate is near 1e-5 shows fewer than one
  error read in expectation. Such rates are not identifiable individually, and the estimate largely reflects the
  population prior. Uncertainty in the error rates is not propagated into the likelihoods.
- The site filter's depth rules cannot distinguish benign sample-specific missingness from mapping
  artefacts, and r5 removes sites without segregating information by design.
- Sample quality exclusion uses private alleles: shared or related donors can evade it, and donors concentrated
  in a few regions can be absorbed by artefact shares. The fitted fraction has a ceiling f h_lo / 2 (about 0.12)
  and is not a calibrated contamination estimate. Local artefacts need not be absorbed. Independent full-filter
  controls flagged 20 dispersed mapping-artefact sites with fractions varying over [0.1, 0.3], where 3.3 kept the
  clean library; another flagged a biased private tract within 1/16 of the contig when most outside cells were
  ineligible (both rules flagged it). There is no general half-contig safety boundary, minimum of 60 clusters,
  or requirement that artefacts share one fraction. Sixteen regions need not supply 16 informative comparisons.
- QC4 reduced clean grid flags from 7 to 0 in 38,313 fits and required-control flags from 33 to 0. Main-grid 2%
  detection rose 8/9 to 9/9 and 5% stayed 9/9, but 1% detection fell 7/9 to 0/9 and combined Q4-fixture 2%
  detection fell 2/2 to 0/2. The heterogeneous 2% cases total 10/11 versus 9/11 descriptively, not as a population
  sensitivity estimate. Repeated controls do not certify a false-positive rate. The N580 seed-2 2% statistic was
  only 22.04 against 20.14. Libraries mostly at 1-5x have little power near the working amplitude.
- Own error above a_min = 0.5% can also trigger quality exclusion. The global Q4 balance interval is not a
  guarantee for each library's private-allele bias. The composite-mixture threshold remains a working choice;
  boundary nuisance parameters, unidentifiable components and weighting prevent a general chi-bar-square
  calibration, which Bonferroni alone cannot restore.
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
the then one-nat model-selection rule (replaced in 2.12). Full regenerated-evidence truth gates are required
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
