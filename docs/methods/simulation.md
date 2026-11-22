# Simulation

Simulation keeps its generating model and truth separate from inference. Explicit named streams preserve generator type, call shape, draw order and contig order. N is generated or observed individuals as stated, L sites, F founder haplotypes and C contigs. The independent-site and fragment read models have different dependence assumptions. Stored pedigree descriptions are truth annotations, never inference metadata.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="ou22"></a>
## ou22 Synthetic example generator

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Write a complete coalescent configuration with an exact count of segregating sites, three diploid cross founders and an explicit effective population size. The seed, contig count, site count and chromosome length flags set its generating design. The example demonstrates software wiring.

Implementation: [interface/example.py](../../haplotype_reconstruction/interface/example.py).

Parameters by name: `EXAMPLE_HAPLOTYPES`, `EXAMPLE_COHORTS`, `EXAMPLE_DEPTH`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: configuration text only; generation occurs at init.

<a id="ou23"></a>
## ou23 Founder materialisation from templates

Validate template axes, finite nonnegative allele masses, increasing positions and a consistent even founder-haplotype count across contigs. Choose the maximum allele probability; for exact ties make one vector draw per haplotype only when ties exist, in site order. Pair consecutive haplotypes into diploid founders. Tie-free rows consume no random draw.

Implementation: [simulation/source.py](../../haplotype_reconstruction/simulation/source.py), [simulation/pedigree.py](../../haplotype_reconstruction/simulation/pedigree.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(H L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou24"></a>
## ou24 Random-cohort pedigree, backcrosses and observation selection

Use the first master draw to seed the structural generator. Build each cohort’s whole pedigree before drawing its contig-by-child seed block. In a backcross-eligible cohort consume the structural probability draw for every child, then select the appropriate ordered parent pair. Select observed samples with the separate observation stream and restore pedigree order. Contigs share structure and scheduling cannot change it; cohort labels remain truth only.

Implementation: [simulation/pedigree.py](../../haplotype_reconstruction/simulation/pedigree.py), [simulation/design.py](../../haplotype_reconstruction/simulation/design.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(total individuals). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou25"></a>
## ou25 Explicit pedigree table

Validate explicit pedigree rows with founders first and distinct earlier parents for each descendant. Simulate by ancestry depth, preserving table order within depth and drawing each contig-by-member seed block before scheduling; this route has no structure-seed draw. Sort recorded gametes by child and slot after all depths. Observation follows the table flag. Sex and generation are stored descriptions read by no inference model; bind table content, not its path.

Implementation: [simulation/pedigree.py](../../haplotype_reconstruction/simulation/pedigree.py), [simulation/design.py](../../haplotype_reconstruction/simulation/design.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: One task set per ancestry depth, with generated haplotypes retained per contig; pedigree validation is linear in rows/edges.

<a id="ou26"></a>
## ou26 Meiosis: gametes, crossovers, mutations, ancestry segments

Draw the starting homologue, then crossover events by the selected map/no-map branch, then mutations. Zero genetic distance or zero fallback rate consumes no crossover draws. Include the final marker with the declared terminal boundary, and assign a marker exactly on a crossover to the new homologue. Simulate maternal then paternal gametes from one child stream. Duplicate mutation indices flip once. Ancestry intervals and fractional map events retain their distinct coordinate forms.

Implementation: [simulation/pedigree.py](../../haplotype_reconstruction/simulation/pedigree.py).

Parameters by name: `MUTATION_RATE_PER_BP`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(L); O(crossovers); O(generated individuals · L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou27"></a>
## ou27 Independent-site read sampler

Let d be configured depth, A_s the existing sample lognormal factor (normalised to sample mean one),
B_i the shared site factor, and G_si an independent cell factor. When enabled,
B_i = exp(Z_i - sigma²/2), Z_i ~ Normal(0, sigma²), sigma² = log(1 + site_depth_cv²),
so E[B_i] = 1 and CV(B_i) = site_depth_cv. With positive site_depth_window_bp,
one B is drawn per occupied fixed physical window floor(position_bp / window_bp), in ascending
window order; otherwise one per site. The population mean is one; no finite-site renormalisation is
applied. Fixed windows supply interpretable local correlation without sequence GC/mappability
covariates or an unsupported smooth covariance law. Hard edges and a common surface across libraries
are approximations; library-specific GC responses are not simulated.

For alpha = count_dispersion > 0, G_si ~ Gamma(shape = 1/alpha, scale = alpha), with
E[G_si] = 1 and Var(G_si) = alpha. Conditional on A_s, B_i and before forced dropout,
D_si ~ Poisson(mu_si G_si), mu_si = d A_s B_i / (1 - w/L), w = floor(L dropout_fraction).
Integrating G gives a negative-binomial marginal with E[D] = mu and Var(D) = mu + alpha mu².
Alpha zero sets G = 1 without draws. Site CV zero sets B = 1 without draws, regardless of window width.
Depth remains d in expectation over factors and dropout; a finite contig need not have exactly that mean.

From the contig's read stream, draw optional normalised sample factors, then optional site factors (both shared
by all samples), then one seed for the per-sample streams. Each sample then draws, from its own stream
`simulation.sample_reads` (that seed and its index): optional gamma factors over its sites, its Poisson depths,
its optional dropout start and its genotype-class binomial ALT counts in class order. Skip a genotype-class draw
when its mask is empty. The independent-site and paired-end samplers use `[simulation.reads] uncalled_base_probability`
to drop each covered base independently. Draw binomial losses for each REF and ALT count at the end of the sample's
stream, after any structural corrections. A rate of zero consumes no draws. At probability p the expected called
depth is (1 − p) times expected covering depth, with the same allele probabilities conditional on a call. Sample
blocks give the same counts at any worker count.

Implementation: [simulation/reads.py](../../haplotype_reconstruction/simulation/reads.py).

Parameters by name: `UNCALLED_BASE_PROBABILITY`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

The opt-in `[simulation.reads] preset = "realistic"` is expanded by the settings loader before mapping
to ReadModel. Explicit TOML/CLI read keys win, including explicit zero values. The resolved config
records the expanded values; see the [user read-model table](../user/quickstart.md#simulation-read-model)
for values, rationale and sources. It sets no inference parameters or target mean depth. With site-depth variation and count dispersion off, their factor draws and arithmetic are skipped;
recorded small-simulation digests cover defaults and paired-end settings.

Cost and scope: O(N L) draws and storage. Per-sample streams allow sample blocks to run independently; shared factors are drawn once per contig.

<a id="ou28"></a>
## ou28 Paired-end fragment read sampler

Draw sample factors, then optional shared site factors once. For each sample draw optional gamma cell
factors, then its fragment count, starts, lengths and chosen homologues. If either new effect is on,
let T_i = B_i G_si. Cut the start axis into windows of the maximum fragment length f_max; a fragment
starting in window k reaches only sites in windows k and k + 1, so M_k = max T_i over those two
windows bounds every target it can cover. Candidate starts in window k form a Poisson process at
M_k times the ordinary rate. Give each fragment one independent Uniform(0,1) mark U; retain its hit
at site i when U < T_i/M_k. Site i's retained intensity is the ordinary rate times T_i whichever window
a covering fragment started in, and candidates scale with local rather than contig-wide maxima of T.
The same mark applies to both mates and all sites, preserving
homologue sharing and nested retention. Conditional depth at site i is therefore Poisson(mu_i G_si),
so integrating gamma factors yields the same negative-binomial count marginal as OU27. The intensity
of hits shared by i and j scales with min(T_i, T_j); it does not depend on an unrelated site's
maximum factor. No independent read top-up or duplication of existing counts is used.
Cost: candidates exceed retained hits by the ratio of local envelope to target, not by the contig maximum.

With the new effects off, skip gamma, envelope and retention draws entirely. The shared factors and the per-sample
stream seed come from the contig's read stream as for independent sites; each sample's fragments, marks, bias,
errors and dropout start come from its own `simulation.sample_reads` stream. Pack first-mate bases before second-mate bases, avoiding double-counted overlap. A sample with no covered base skips later base draws. Apply optional heterozygote bias, then the declared homozygous error mask, accumulating counts by site. Fragment correlation is part of the generating model.

Implementation: [simulation/reads.py](../../haplotype_reconstruction/simulation/reads.py).

Parameters by name: `FRAGMENT_QUANTILES`, `MINIMUM_FRAGMENT_BP`, `FRAGMENT_SD_LIMIT`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N L + candidate fragments + candidate site hits), with per-sample temporary
storage for candidate fragments/hits. With new effects enabled, the maximum propensity M multiplies
candidate intensity; extreme site CV or dispersion can increase time/memory substantially. The
sampler has no truncation cap on these factors. The default path retains its previous costs.

<a id="ou29"></a>
## ou29 Truth records: alleles, painting chunks, crossovers, manifest, truth pedigree

Publish observations and independent truth with aligned sample/site axes, true founder alleles, observed-sample alleles, ancestry chunks and analysed-child gametes. Truth chunks use the union of homologue segment bounds and require whole-chunk containment. Stored integer bounds truncate fractional events, a known truth-surface limitation that can disagree at a boundary marker; crossover positions retain float precision. Truth metadata never enters inference.

Implementation: [simulation/source.py](../../haplotype_reconstruction/simulation/source.py), [simulation/truth.py](../../haplotype_reconstruction/simulation/truth.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N · segments). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou30"></a>
## ou30 Simulation source: streams, identity, resume, scheduling and memory

Bind the realised seed once and preserve named founder, pedigree, observation, gamete and read stream constructors and draw order. Materialise the whole configured contig set, reusing only compatible completed source products. Publish truth before its read-completion product. Keep generated and analysed sample axes distinct and reject partial products without their manifest. Task order may change execution, never per-contig seeds or data. Each contig's reads are simulated as
blocks of samples on all CPUs from shared read-only inputs, and the contig's records are written by a gather on
its own worker; meiosis tasks carry only the parents their children use.

Implementation: [simulation/source.py](../../haplotype_reconstruction/simulation/source.py), [stages/inputs/simulation.py](../../haplotype_reconstruction/stages/inputs/simulation.py).

Parameters by name: `READS_BYTES_PER_CELL`, `READS_WORKER_BYTES`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Sum of generation and read-sampling costs; whole-contig read arrays often dominate memory.

<!-- named mechanism anchors -->

## Named mechanisms

<a id="example"></a>

**Example:** see [OU22](#ou22).



## Sources, stocks and segregating sites

A coalescent source samples two stable source IDs per generation-zero stock individual and one reference leaf. Stocks share their pedigree and selected last-generation individuals across contigs; mutation-free label meioses use independent stock streams. The cross assigns a separate founder-haplotype axis to these selected mosaics.

The structured SMC′ genealogy is traversed twice with its Generator state restored. The first pass measures branch length separating the carried sources; the second places mutations at the independently selected integer bp sites. An intensity profile multiplies separating length. Count mode chooses the smallest exponential clocks, per-Mb mode uses a finite clock threshold, and supplied positions with zero separating length are recorded as dropped. Flat map intervals receive sites. The reference leaf determines REF coding and is never carried by a cross founder.

Source IDs are stable across contigs. Stored source rows include only IDs carried at retained markers; the row-to-ID mapping accompanies every panel. Stock ages share the source generation-zero clock, an approximation for histories short relative to source coalescence.

Implementation: [stocks.py](../../haplotype_reconstruction/simulation/stocks.py), [source.py](../../haplotype_reconstruction/simulation/source.py), [genealogy.py](../../haplotype_reconstruction/simulation/genealogy.py), [sites.py](../../haplotype_reconstruction/simulation/sites.py).

Structural input errors are checked before draws. The source, stock, site and cross steps of all pending contigs run concurrently as single-core tasks; each contig's read and truth arrays then belong to one engine lane and are released when its truth and reads are published. Results do not depend on the number of CPUs or on task order. The released cross seed schedule is replayed over the full configured axis; source, profile, site and stock streams depend on contig identifiers.

## Read loss, regional depth and call-set sites

These options are off by default; a configuration that does not name them draws exactly the reads and sites it drew before.

**Shared read loss at two scales.** With `site_loss_fraction` f1 > 0, each occupied window of `site_depth_window_bp` (each site when the width is zero) is impaired independently with probability f1, and an impaired window keeps the share 1 - L of its reads, L ~ Uniform(0, 1). With `site_loss_fine_fraction` f2 > 0 the same law acts independently on the windows floor(position_bp / `site_loss_fine_window_bp`). Both multiply the lognormal site factor B_i, are shared by all samples and are used by both samplers. Loss is not renormalised: `depth` is the unimpaired depth, and the expected mean depth is depth (1 - f1/2)(1 - f2/2). The draws follow the lognormal draws on the contig's read stream, coarse then fine; a zero fraction consumes no draw. Two scales give a site-depth correlation that falls quickly over a few hundred bp and slowly over kilobases, as real call sets show; one window width gives a single linear decay.

**Regional depth per sample.** With `region_depth_cv` > 0, every sample has an independent mean-one lognormal factor per occupied window of `region_depth_window_bp`, drawn from that sample's own regional stream and multiplied into its site intensities before its reads are sampled. It represents the slow, library-specific variation of depth along a chromosome.

**Sites of a call set.** `[simulation.sites] variant_against = "reference"` places sites on the branches that separate the carried sources or separate them from the reference haplotype, so positions where the whole cohort differs from the reference appear at their genealogical rate; the default `"cohort"` keeps only sites that segregate among the cross's founders. `spurious_fraction` = s adds positions with no true variant: with an exact `count` they are round(s x count) of the total, otherwise round(g s / (1 - s)) for g genealogical sites; they are drawn by a second race with the profile's weights alone, carry the reference allele in every source, and are marked in the truth (`spurious`).

The preset `short_read_cohort` and the reference configuration `configs/cohort_callset.toml` set these, the heterozygote balance of the artefact layer and the paired-end sampler to resemble a 290-sample short-read cohort at a median depth of about 7.5. The generated statistics vary between realisations: on 3 Mb the site-depth correlation at 150-400 bp differs between seeds by 0.1 or more, so compare several seeds rather than one.

Implementation: [reads.py](../../haplotype_reconstruction/simulation/reads.py), [sites.py](../../haplotype_reconstruction/simulation/sites.py). Parameters and units are in the [settings reference](../reference/settings.md).

## Call-set artefacts

The optional reference_blocks preset transforms sampled counts with independent site and sample streams. Heterozygous cells receive beta-binomial balance; systematic sites replace hom-reference ALT errors; non-reference misses combine hard-site, renewal-stretch and sample terms; uncalled cells form constant logarithmic-depth-band blocks reporting their minimum depth and no ALT reads. Regions multiply the three miss terms before clipping. Truth stores missed, block and systematic-site masks before the reads completion record.

A caller misses a homozygous-ALT genotype far less often than a heterozygote, since all its reads agree: the miss
probability of a dosage-2 cell is `hom_alt_miss_factor` times a heterozygote's (the same uniform decides, so a factor
of 1 is the 4.15 law). The preset's 0.02 is the ratio measured in the 52-sample ONT trios (0.04-0.11% of hom-ALT cells
missed against 3.5-4.6% of heterozygotes); at a factor of 1 a missed hom-ALT parent also looks hom-REF and turns
sites into false must-heterozygous sites. Short-read call sets made by joint pileup calling write every sample's own
read counts and have no reference blocks; their misses are not this layer's (the shipped short-read configuration
uses only `balance_dispersion`).

`balance_dispersion` redraws each heterozygous cell's allele counts on their own. With long reads
(`long_read_mean_bp` > 0) it must be 0, and the settings refuse another value: the read model already makes the
heterozygote balance (context sites, reference bias), and a per-cell redraw would erase the correlation that reads
shared by neighbouring sites give their allele depths. The `long_read_cohort` read preset sets it to 0, so it combines
with `reference_blocks`. The paired-end sampler keeps the per-cell term.

Implementation: [artefacts.py](../../haplotype_reconstruction/simulation/artefacts.py). Parameters and units are in the [settings reference](../reference/settings.md).

## Long-read sampler

With `long_read_mean_bp` > 0 the reads are single long reads instead of independent sites or paired-end fragments
(the other samplers and their draws are unchanged). Read starts form a Poisson process along the contig at the rate
that gives every site an expected `depth` reads; a read's length is lognormal with mean `long_read_mean_bp` and
log-scale SD `long_read_length_sigma` (clipped to [`LONG_READ_MIN_BP`, exp(log-mean + `LONG_READ_SIGMA_LIMIT` SD)]),
it comes from one homologue with probability 1/2 and covers every site it spans. Sample depth factors, shared site
factors, regional factors and gamma cell factors act through the same envelope thinning as for fragments.

Each read draws one error rate e = 10^(-Q/10), Q ~ Normal(`read_quality_mean`, `read_quality_sd`) clipped to
[`READ_QUALITY_MIN`, `READ_QUALITY_MAX`], and uses it at every site it covers, so a poor read errs at many nearby
sites (errors correlated within a read). At a covered site the read's base is lost to a one-base deletion
(e (1 - `substitution_share`) / 2) or a substitution to a third base (2 e `substitution_share` / 3), or reads the other
allele (e `substitution_share` / 3). Three mechanisms act on SNP calls beyond base errors:

- **Context sites.** A share `context_sites` of sites, the same in every sample, has one allele that extends a
  homopolymer (drawn per site). Reads carrying it suffer an extra indel at the site with probability
  min(1, e (1 - `substitution_share`) `context_factor` s), s a mean-one exponential strength of the site (runs and
  their sequence differ); `context_convert` of those read as the other allele (the aligner places the run boundary
  on the other base) and the rest give no base. This gives site-specific allele
  bias shared by every sample.
- **Reference bias.** A read's base is lost (local clipping or misalignment) with probability
  1 - exp(-`reference_bias` k^`reference_bias_power`), k the ALT alleles of the read's homologue within
  `reference_bias_window_bp` of the site (the site included): reads of a homologue that differs densely from the
  reference lose bases, and adjacent sites of one read share k, so the loss is correlated within the read.
  `reference_pull` reads an ALT base as REF with mean probability `reference_pull` (scaled by e over its mean).
- **Foreign reads.** A share `foreign_read_share` of a sample's reads come from a uniform other sequenced sample and
  homologue (cross-sample contamination or barcode misassignment); their alleles enter the sample's counts.

The causes combine as independent hazards. Only the allele depths reach the pipeline: what a long read shows beyond
them (read-backed phase, the read's identity across sites, base qualities, indel and multi-allelic records,
supplementary alignments) is not written and the pipeline could not read it. The correlation of allele depths
between nearby sites, the per-read error clustering, the site-shared allele bias and the haplotype-level reference
bias are what the pipeline's inputs can reflect, and they violate its working assumption that sites are independent
given the genotypes. `error_rate` is not used by this sampler. Context sites are drawn on the contig's read stream
after every released draw; each sample's reads, lengths, homologues, qualities, foreign donors, marks and outcomes
come from its own `simulation.sample_reads` stream.

The `long_read_cohort` read preset holds values calibrated on a 52-sample ONT cohort called with Clair3 and
GenotypeGVCFs (six trios inferred from the data; one estimator applied alike to real and simulated trios, on
heterozygous cells that show both alleles: allele fraction and depth correlation by distance, heterozygote balance
by local ALT density, allele bias shared between trios, background error rates and within-sample error
co-occurrence). Its read length is the effective length of the reads that reach the allele depths (about 3 kb
mean), not the library's. Cells with no ALT or no REF read at a true heterozygote (3-5% and 0.3-0.7% of such cells in
that cohort) are call-set misses: add the call-set artefact preset for them and for reference blocks, with
`balance_dispersion = 0`, since the read model already gives heterozygotes their spread (and that layer redraws each
heterozygous cell on its own, which would erase the correlation the shared reads give neighbouring sites).

Implementation: [reads.py](../../haplotype_reconstruction/simulation/reads.py). Parameters by name:
`LONG_READ_MIN_BP`, `LONG_READ_SIGMA_LIMIT`, `READ_QUALITY_MIN`, `READ_QUALITY_MAX`; settings and units in the
[settings reference](../reference/settings.md).

Cost and scope: O(N (reads + covered sites)); reads of mean length m cover about m x site density sites each.

## Structural rearrangements

`[simulation.structure]` (off by default; preset `cichlid`) makes the simulated genomes structurally different from
the reference whose coordinates the VCF uses. Every variant is a half-open reference interval [start, end):

- `inversion`: carriers hold the interval in the opposite orientation;
- `deletion`: carriers lack the interval (their cells are hemizygous or absent; no reads come from that homologue);
- `duplication`: carriers hold an extra copy at `target` (tandem when `target` = end). The copy is absent from the
  reference, so its reads align to the original interval: depth rises with copy number and the copy's paralogous
  differences (`paralog_divergence` of its sites) appear as extra alleles in carriers;
- `reversed`, `moved`, `unlinked`: the reference order is wrong for every haplotype; the segment truly lies
  reversed in place, between `target - 1` and `target`, or on another chromosome.

Variants are explicit (`variants`, with optional carriers by source haplotype index) or random per contig: large
inversions (Poisson `inversion_rate` per contig, span a uniform fraction [`inversion_span_lo`, `inversion_span_hi`] of
the contig), misordered segments, small inversions, deletions and duplications (Poisson per Mb, exponential lengths);
a candidate overlapping an accepted one is rejected (order variants never overlap, nor do deletions and
duplications; a deletion or duplication may lie inside an inversion). Carriers are stable source haplotypes: with a
coalescent source a variant is a mutation on the source genealogy at its anchor (start; a duplication's target),
replayed on the same trees as the sites and coded relative to the reference leaf (an inversion the reference carries
reverses who is inverted; a deletion or duplication the reference carries changes no reference-axis call and has no
carrier); with a file source each source haplotype carries it with probability `carrier_frequency`. A haplotype
carries a variant when its source at the anchor does.

Meiosis keeps the released crossover draws and filters and orders them (stocks and cross alike):

- a deletion carried by either homologue removes the crossovers inside it;
- an inversion carried by exactly one homologue (a heterokaryotype) keeps the crossovers inside it only in even
  number: a chromatid with an odd number of exchanges in the inversion loop is dicentric or acentric and gives no
  viable gamete, a chromatid with an even number is balanced; the viable product of an odd draw is the
  non-recombinant one;
- the transmitted chromatid follows the physical order of the parent's chromosomes: homokaryotypic inversions and
  reversed segments are traversed backwards, moved segments sit at their targets, unlinked segments start from their
  own homologue choice (stream `simulation.sv.gamete`). One crossover inside a homokaryotypic inversion therefore
  changes the reference-axis haplotype three times, at start, at the crossover and at end.

The generating meiosis has no crossover interference, so double exchanges inside a long heterokaryotypic inversion
appear at Poisson frequency, more often than in fish with strong interference; gene conversion is not modelled.
The source genealogy has no arrangement-specific suppression, so inverted and standard source haplotypes are not more
diverged inside an inversion than elsewhere.

Truth (`structure` in the contig truth, absent when off) holds the variants, the carriers of every cross-founder and
analysed haplotype, the paralogous sites and, per gamete, the reference-axis switch points (equal to the crossovers
except around rearrangements). The crossovers in truth are the realised meiotic exchanges. Evaluation scores a
sample's cells inside a deletion it carries as uncalled and treats founder alleles inside a carried deletion as not
represented, and adds `called_intervals_at_rearrangement_switch` to the map columns. Read draws for copies and for
deletions under the independent-site and fragment samplers use stream `simulation.sv.reads` per sample (those
samplers do not know which homologue a read came from, so each read is assigned at 1/2 and the absent homologue's
reads removed); the long-read sampler drops an absent homologue's hits directly. Copy reads are independent per site.

The `cichlid` preset: large inversions on five of 22 chromosomes, each 17-23 Mb and more than half the chromosome,
polymorphic among lineages and with crossovers suppressed in heterokaryotypes (Blumer et al. 2025, Science 388,
eadr9961); deletions and duplications of transposable-element size at the order of the Lake Malawi pangenome's
variable sequence (Quah et al. 2024, bioRxiv 2024.03.28.587230); reference misorder at about 12 segments per Gb, from
shared linkage breakpoints in a long-read assembly's cross. None of these values was fitted to a cross.

Implementation: [structure.py](../../haplotype_reconstruction/simulation/structure.py),
[pedigree.py](../../haplotype_reconstruction/simulation/pedigree.py),
[stocks.py](../../haplotype_reconstruction/simulation/stocks.py). Settings and units are in the
[settings reference](../reference/settings.md).

Cost and scope: one extra genealogy traversal per contig with a coalescent source; meiosis O(segments x crossovers)
per gamete; read arrays grow by two int8 and two bool copies of the sample-site axis.

## Injected outliers (gate G1i)

A validation option (`injected_outlier_rate`, `injected_outlier_pattern`; 4.21, ADVICE2 11.4): for sample n and site
s, with probability r, the ORDERED genotype of the sample at the site is replaced by a draw from the ordered cohort law
(q0; q1/2, q1/2; q2) of the simulated cohort at the site, before any read is drawn. The independent-site sampler draws
its counts from the replaced genotype; the fragment sampler keeps its fragments, their homologue assignment and
depths, and every fragment covering the site reads the replaced allele of its own homologue (cross-site allele noise
preserved). A redraw that returns the same genotype is part of the model. Patterns at the setting r: uniform;
heterogeneous (each sample heavy with probability about one in ten, by a hash of the sample identifier, not a fixed
quota; heavy samples at 10 r_low and the rest at r_low, r_low = r / 1.9, so the cohort mean is r only when exactly a
tenth is heavy: with H heavy of N it is r (1 + 9 H / N) / 1.9; r at most 0.19); clustered (runs of five consecutive
sites starting with probability r / 5 per sample and site, each site of a run replaced by its own draw; away from the
ends a site is replaced with probability 1 - (1 - r / 5)^5, slightly below r); misspecified (half of the replacements
forced heterozygous). Each sample draws from its own stream (`simulation.injected_outliers`) and the law is the whole
analysed cohort's, so the cohort is injected in one call; a rate of zero draws nothing. With the independent-site
sampler a replaced genotype also changes the random assignment of other cells' allele counts (their depths stay), so
two runs of one seed differ outside the mask as well; the fragment sampler keeps the other cells' observations. Truth keeps the original alleles (ancestry), the replaced alleles and the mask. Two readings,
never pooled: (R) the replacement is a read artefact, every evaluation against the original alleles; (M) it is a real
deviation, founders and ancestry against the original, released genotypes and phase against the replaced alleles.

Implementation: [simulation/injection.py](../../haplotype_reconstruction/simulation/injection.py),
[simulation/source.py](../../haplotype_reconstruction/simulation/source.py).
