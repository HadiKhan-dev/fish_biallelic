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
its optional dropout start and its genotype-class binomial ALT counts in class order. `heterozygote_alt_probability`
belongs to this sampler only: it has no reads, so a per-cell heterozygote balance is its whole law (the read samplers
refuse a value other than 0.5; their imbalance is read-level, [OU28](#ou28)). Skip a genotype-class draw
when its mask is empty. The independent-site and paired-end samplers use `[simulation.reads] uncalled_base_probability`
to drop each covered base independently. Draw binomial losses for each REF and ALT count at the end of the sample's
stream, after any structural corrections. A rate of zero consumes no draws. At probability p the expected called
depth is (1 − p) times expected covering depth, with the same allele probabilities conditional on a call. Sample
blocks give the same counts at any worker count.

Implementation: [simulation/reads.py](../../haplotype_reconstruction/simulation/reads.py).

Parameters by name: `UNCALLED_BASE_PROBABILITY`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

The opt-in `[simulation.reads] preset = "realistic"` (and `"realistic_long_reads"`) is expanded by the settings
loader before mapping to ReadModel; see [Realistic profile](#realistic-profile). Explicit TOML/CLI keys win, including
explicit zero values. The resolved config records the expanded values. It sets no inference parameters or target
mean depth. With site-depth variation and count dispersion off, their factor draws and arithmetic are skipped;
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
errors and dropout start come from its own `simulation.sample_reads` stream. Pack first-mate bases before second-mate bases, avoiding double-counted overlap. A sample with no covered base skips later base draws. Apply the declared homozygous error mask, accumulating counts by site. Fragment correlation is part of the generating model.

**Reference mapping bias (read-level, 4.22).** Allelic imbalance is a property of reads, not of sites: a read that
differs from the reference at several positions fails the aligner's mismatch or mapping-quality filters as a whole.
Each mate fails to align, and loses every base it covers, with probability

  1 - exp(-`reference_bias` k^`reference_bias_power`),

where k is the number of ALT alleles of the fragment's homologue at the sites under that mate's own span, so reads of
a haplotype that differs densely from the reference lose most: the bias depends on the lineage a read comes from. A
read with no ALT allele is never lost. A lost first mate leaves the overlap to the second mate, which then reads it.
One uniform per mate decides, drawn from the batch's stream after the homologues and before the thinning marks; with
`reference_bias` zero nothing is drawn and the sampler draws the same reads as without it. A kept read therefore
shows its homologue's alleles at every site it covers. This replaces the per-hit conversion of 4.21
(`heterozygote_alt_probability` converted the disfavoured allele independently at every heterozygous site a read
covered, so one read could disagree with its own homologue at some sites and not others; 4.21 D5 showed the read
state reading those as phase-inconsistent reads). Depth falls with the lost mates: configured depth is before the loss.

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

The preset `short_read_cohort` and the reference configuration `configs/cohort_callset.toml` set these and the paired-end sampler to resemble a 290-sample short-read cohort at a median depth of about 7.5. Their heterozygote balance (a per-cell ALT probability of 0.48 and the artefact layer's per-cell redraw of 0.026 until 4.21) is now the read-level mate mapping loss (`reference_bias` 0.07, power 2), not refitted to that cohort. The generated statistics vary between realisations: on 3 Mb the site-depth correlation at 150-400 bp differs between seeds by 0.1 or more, so compare several seeds rather than one.

Implementation: [reads.py](../../haplotype_reconstruction/simulation/reads.py), [sites.py](../../haplotype_reconstruction/simulation/sites.py). Parameters and units are in the [settings reference](../reference/settings.md).

## Call-set artefacts

The optional call-set layer (`[simulation.artefacts] enabled = true`, or `preset = "reference_blocks"` for the
defaults) transforms sampled counts with independent site and sample streams, in model order:
1. Heterozygous cells receive beta-binomial balance (`balance_dispersion`, the independent-site sampler only, below).
2. A share `systematic_sites` of sites are error-prone, each with a per-read rate drawn from Uniform(`systematic_lo`,
   `systematic_hi`) and shared by every sample: at such a site each REF read is read as the ALT base with that rate,
   whatever the cell's genotype (a context error, or low-level mismapping from a similar locus). Every cell keeps its
   depth and its other ALT reads (contamination, paralogue copies).
3. Missed variant calls: a variant cell (heterozygous or homozygous ALT) is missed with a probability that combines
   hard-site, renewal-stretch and per-sample terms (`hard_sites`/`hard_miss`, `stretch_cover`/`stretch_sites`/
   `stretch_miss`, `miss` with a lognormal sample factor of CV `miss_sample_cv`; `regions` multiply the three terms
   before clipping).
4. A missed cell is reported as a reference block: runs of adjacent missed cells of a sample within one logarithmic
   depth band (`block_band`) report the run's minimum depth and no ALT read.

**Which counts can be rewritten: missed cells only** (owner, 4.22). A missed call is a caller effect: the caller
reports the sample reference-homozygous at a variant, and a gVCF-style call set then carries a reference block there.
No other cell joins a block: a true reference-homozygous cell keeps its own REF and ALT counts, stray ALT reads
included, whatever their share (there is no minimum ALT fraction for a reference call). Nothing else in the simulator
sets counts to zero or lowers them because of a cell's genotype or allele fraction, with one redraw: at a true
heterozygote `balance_dispersion` redraws the ALT/REF split at the cell's own depth (the independent-site sampler's
overdispersed law; it deletes nothing, is refused with reads, and no realistic preset uses it). Reads lost before
counting are physical and act whatever the genotype (whole mates lost at a divergence-dependent rate,
`reference_bias`; deletions; dropout tracts; depth loss). Truth stores the missed-cell and systematic-site masks before the reads
completion record.

A caller misses a homozygous-ALT genotype far less often than a heterozygote, since all its reads agree: the miss
probability of a dosage-2 cell is `hom_alt_miss_factor` times a heterozygote's (the same uniform decides, so a factor
of 1 is the 4.15 law). The default 0.02 is the ratio measured in the 52-sample ONT trios (0.04-0.11% of hom-ALT cells
missed against 3.5-4.6% of heterozygotes); at a factor of 1 a missed hom-ALT parent also looks hom-REF and turns
sites into false must-heterozygous sites.

`balance_dispersion` redraws each heterozygous cell's allele counts on their own. With reads (`read_length_bp` > 0
or `long_read_mean_bp` > 0) it must be 0, and the settings refuse another value: a per-cell redraw makes the reads
covering the cell report the other homologue's allele there and their own elsewhere, a within-read inconsistency no
molecule has (the reason the per-hit heterozygote balance was retired, 4.21 D5), and it erases the correlation that
shared reads give neighbouring sites. Reads get their balance from `reference_bias`. The redraw remains for the
independent-site sampler. The `long_read_cohort` and realistic read presets set it to 0.

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
- **Reference bias.** A read's base is lost (local clipping or misalignment; a long read is not lost whole for a few
  mismatches, unlike a paired-end mate) with probability
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
that cohort) are call-set misses: add the call-set artefact preset for them (missed cells are reported as reference
blocks), with `balance_dispersion = 0`, since the read model already gives heterozygotes their spread (and that layer
redraws each heterozygous cell on its own, which would erase the correlation the shared reads give neighbouring
sites).

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

### Collapsed paralogs

`collapsed_fraction` > 0 (with `enabled = true`; off by default) adds segmental duplications that the reference
assembly collapsed into one segment, so the reads of every copy align to that segment. The ordinary `duplication`
above is a copy absent from the reference in some haplotypes; a collapsed paralog is a copy that the reference
itself merged, so it can be shared by every sample.

- **Segments**: disjoint half-open intervals, a Poisson number with mean `collapsed_fraction` x contig length /
  `collapsed_mean_bp`, uniform starts, exponential lengths (a segment overlapping an earlier one is dropped).
- **Copy number**: each stable source haplotype carries a segment's copy with probability
  `collapsed_carrier_frequency`. At 1 the copy is fixed and every sample has two extra copies. Below 1 it is a
  copy-number polymorphism inherited through the pedigree: a sample has 0, 1 or 2 extra copies, read from each
  homologue's source haplotype at the segment's start.
- **Paralogous sequence variants (PSVs)**: a share `collapsed_psv_share` of the written sites inside a segment are
  sites the cohort does not segregate. The panel sets them to REF in every haplotype and marks them spurious, so the
  truth calls them monomorphic. The copy carries ALT there.
- **The cohort's own SNPs inside a segment**: the copy carries the ancestral base (REF, or ALT where the reference
  genome holds the derived allele), because the SNP arose at the other locus.
- **Reads**: each carried copy adds one homologue's depth of reads inside the segment, times the sample's site and
  regional depth factors (a near-identical copy shares the segment's mappability; read geometries are drawn at the
  segment's largest factor and each read's hits kept with one uniform per read, as the fragment sampler does). They
  follow the sampler's geometry (independent sites, paired mates, long reads), are clipped to the segment, carry the
  copy's base, undergo the same read-level mapping loss as own reads (k counted on the copy's alleles) and carry no
  sequencing error. Neither the count dispersion nor the dropout tract acts on them (both off by default): a dropout
  tract inside a segment shows copy reads only, so its PSVs look homozygous ALT. Stream `simulation.sv.collapsed`.
  Truth `structure` records the segments, every sample's copies and the PSV sites (`collapsed_start`,
  `collapsed_end`, `collapsed_copies`, `collapsed_psv_sites`).
- **Linkage**: copy number follows the source haplotype at the segment's start, so it is completely linked to the
  local founder haplotype and inherited with it (tandem-like). A dispersed real paralog would segregate independently
  of the local haplotype; the model's depth signal is therefore heritable and founder-linked, which may be friendlier
  to inference and to QC's r2 than reality. Each source haplotype carries the copy independently, also under a
  coalescent source (polymorphic structural variants use the genealogy instead). The paralog has one fixed sequence
  per contig: no polymorphism at the paralog and no gene conversion. `collapsed_fraction` is the expected share
  before overlaps are dropped and segments clipped at the contig end (the realised share is about 3% lower).

A PSV site in a sample with two extra copies shows ALT in half of its reads at about twice the depth: a
pseudo-heterozygote shared by every sample. A cohort heterozygote inside a segment shows its ALT allele in a quarter
of the reads. Site QC's depth rule removes most of both.

**Calibration** (f3-like design under the realistic preset, 11x, three contigs; `lab422/sim` BUILD.md sections 11,
12 and 16).
- TroMau's targets:
  - sites that look heterozygous in at least 80% of samples: 0.0145;
  - of those, sites whose mean depth is also above 1.5x the median (paralog-like): 0.0068;
  - all sites above 1.5x the median: 0.038;
  - after site QC, kept sites that look heterozygous in at least 80% of samples: 4e-4.
- The realistic preset without collapsed paralogs gives 1e-4, 7e-5, 0.003 and 3e-5.
- The kept-site share comes first, because the pipeline reads kept sites. The preset's values are
  `collapsed_fraction` 0.055, 5 kb segments, `collapsed_psv_share` 0.02 and `collapsed_carrier_frequency` 0.8 (young
  duplications with few fixed differences). They give 0.0078, 0.0052, 0.036 and 4.3e-4, all four inside the frozen
  tolerances.
- Raising the PSV share raises the pre-QC shares toward TroMau's, but the kept share rises with it (PSV share 0.13:
  0.0112, 0.0073 and 9.4e-4).
- The pre-QC heterozygote ALT share falls from 0.478 to 0.464 (TroMau 0.448), while the QC-kept share stays 0.487.
  These sites are biased and then removed, as in TroMau.
- Site QC removes 95.5% of the PSV sites (47 of 1,046 survive) and 99.0% of the cohort SNPs inside segments (475 of
  49,161 survive). The depth rule r1 flags about 89%, the sample-depth rule r2 about 25% and the informativeness rule
  r5 about 50%.
- The survivors are copies thinned in low-mappability windows, whose depth stays near normal.
- With fixed copies (`collapsed_carrier_frequency` 1), r2 flags nothing on some contigs: its artefact class does not
  separate from the clean core, which TroMau's QC does not show (r2 removes 36% of its sites). An inherited
  copy-number polymorphism keeps r2 working.
- In TroMau about half of the sites that look heterozygous in at least 80% of samples have normal depth. Collapsed
  copies at unimpaired sites raise depth, so most of those need another cause, which is not modelled; one example is
  cross-mapping between paralogs that the assembly keeps apart.

Implementation: `CollapsedParalogs`, `draw_collapsed`, `collapsed_psv_sites`, `collapsed_copies` in
[structure.py](../../haplotype_reconstruction/simulation/structure.py); the PSV panel and the copies in
[source.py](../../haplotype_reconstruction/simulation/source.py); the copy reads in
[reads.py](../../haplotype_reconstruction/simulation/reads.py). Cost: one pass over each carried segment's reads per
sample, about collapsed_fraction x 2 of the sample's reads.

<a id="exogenous"></a>
## Exogenous (other-species) contamination

`[simulation.reads] exogenous_read_share` > 0 (default 0: nothing is drawn and every configuration draws the same
reads) adds reads of another species, for example human DNA from handling. Most such DNA never maps; the reads that do
land where the reference is conserved enough to align them, so they are confined to conserved windows: a share
`exogenous_window_fraction` (default 0.03; about 3% of a vertebrate genome is under strong constraint) of the occupied
windows floor(position / `exogenous_window_bp`) (default 500 bp), shared by all samples and drawn from the contig's
read stream after every other draw. Inside them each site carries the other species' base, fixed per site (one foreign
genome) and independent of the cohort: ALT with probability `exogenous_alt_probability` (default 0.2: in a conserved
window the other species carries the ancestral base, and at a cohort SNP the ALT allele is the ancestral one at a
minority of sites, the reference holding the derived one), REF otherwise; third bases are not modelled.

A sample is contaminated with probability `exogenous_sample_fraction` (default 1), drawn once per sample from the
stream `simulation.exogenous_level` keyed by the generating seed and the sample identifier, so the same sample has the
same share on every contig. Its share of mapped reads is `exogenous_read_share` x Exponential(1) (libraries differ
widely in handling contamination), clipped to `EXOGENOUS_SHARE_LIMIT`. Mapping filters are part of the share: it is
the share that maps. All of a sample's exogenous reads fall in the conserved windows, so their depth there is
depth x sample factor x s / (1 - s) x (span of the sites / conserved bp): at s = 0.02 and 3% of windows about two thirds
of the endogenous depth. Reads follow the sampler's geometry (paired mates, long reads, or independent sites for the
independent-site sampler); a read overlapping a window keeps only its part inside it (it fails to align beyond), and
starts are drawn so that every window position has the same expected depth. They carry no sequencing error and come
from the sample's own stream `simulation.exogenous_reads` after its other draws (before the uncalled-base loss, which
acts on them too).

What the pipeline can see: extra depth and a minority allele at the windows' sites, the same allele in every
contaminated sample and at a share that varies between samples, and nothing elsewhere. Unlike cross-sample
contamination (`foreign_read_share`), the alleles do not follow any sequenced sample's haplotypes.

Implementation: [reads.py](../../haplotype_reconstruction/simulation/reads.py) (`_exogenous_counts`,
`exogenous_shares`), [source.py](../../haplotype_reconstruction/simulation/source.py). Parameters by name:
`EXOGENOUS_READ_SHARE`, `EXOGENOUS_SAMPLE_FRACTION`, `EXOGENOUS_WINDOW_FRACTION`, `EXOGENOUS_WINDOW_BP`,
`EXOGENOUS_ALT_PROBABILITY`, `EXOGENOUS_SHARE_LIMIT`. Off in every preset.

Cost and scope: O(exogenous reads + covered window sites) per contaminated sample.

<a id="realistic-profile"></a>
## Cross-sample reads and contamination

Two settings put another individual's molecules into a library. Both are off by default and draw nothing then.

**Uniform foreign reads** (`foreign_read_share`; long reads and paired-end fragments)
- Each read, or each fragment with both mates, comes with that probability from a uniform other sequenced sample's
  homologue (barcode misassignment, index hopping, low-level bleed).
- The foreign molecule uses the sampler's own geometry and depth thinning, and the read-level mapping loss with k
  counted on the donor homologue's alleles. Its errors follow the donor's genotype.
- The independent-site sampler refuses the setting.

**Per-library contamination** (every sampler): a contaminated library takes each read from its **one donor** with
probability its share. Heavy contamination and sample mix-ups usually involve a single other individual, so the
donor is fixed per library, genome-wide. Two ways to say who and how much:
- **A law:** a library is contaminated with probability `contamination_sample_fraction`, at `contamination_share` x
  Exponential(1) clipped at 1 (1 = a mix-up). The donor is drawn uniformly among the other **sequenced** samples
  (stream `simulation.contamination_level`, keyed by the generating seed and the sample identifier).
- **An explicit list:** `contamination_samples`, `contamination_shares` and optionally `contamination_donors` (a
  uniform other sequenced sample each when omitted). The list replaces the law. A listed donor may be an unsequenced
  individual of an explicit pedigree, for example an outside fish added as an unobserved founder without offspring.

How contamination is drawn:
- Depth is unchanged: contamination replaces a library's reads. Uniform foreign reads are drawn first, exactly as
  the long-read sampler always drew them.
- Only the donor's two haplotype rows enter a read block.
- Where the recipient carries a deletion, only its own reads are removed: the paired-end and long-read samplers drop
  own hits at absent homologues and keep donor hits. The independent-site sampler replaces reads after the structure.
- The recipient's collapsed-paralog copy reads are scaled by (1 - share), and the donor's own collapsed copies add
  reads at the share.
- Not applied on the donor side: the donor's deletions and its structural duplication copies.
- The truth records each library's realised share and donor (`contamination_share`, `contamination_donor`).

**Sample QC.** Sample QC's contamination test looks for private alleles: a cell counts only when no other library
carries the minor allele. A donor that is itself sequenced is therefore invisible to it, whatever the level, and the
law's donors are always sequenced. Use an explicit list with an unsequenced donor to exercise the test.

Implementation: `contamination_levels`, `_read_donors`, `_donor_alleles` and the samplers in
[reads.py](../../haplotype_reconstruction/simulation/reads.py); per-library shares and unsequenced donors in
[source.py](../../haplotype_reconstruction/simulation/source.py).

## PCR duplicates

`duplicate_rate` (paired-end only; off by default) leaves PCR duplicates in a library's reads, as when a pipeline does
not mark them. A pipeline that marks and removes duplicates sees only a lower depth, which the depth law already
covers, so the model is the unmarked case.
- **Rate:** a library's rate d is `duplicate_rate` x Exponential(1), clipped at 0.9 (stream
  `simulation.duplicate_level`, keyed by the generating seed and the sample identifier). The truth records it
  (`duplicate_rate`).
- **Copies:** each sequenced molecule has Poisson(d / (1 - d)) duplicates, so duplicates are a share d of the reads.
  Depth is the configured depth of unique fragments plus the duplicates.
- **What a duplicate repeats:** its parent's mapping, depth retention and alleles, including the parent's errors (a
  PCR or sequencing error in the parent molecule repeats). It also adds its own independent errors. The effect is
  extra depth with correlated alleles: heterozygote allele shares and wrong-allele counts are overdispersed.
- **Every molecule of the library is duplicated:** own fragments, donor fragments, collapsed-paralog copy reads,
  exogenous reads and structural copy reads. Own reads leave a carried deletion inside the sampler, so duplicates
  keep their correlation there.
- **Refused** with long reads (PCR-free long-read libraries carry few duplicates; an amplified long-read library is
  not modelled) and with the independent-site sampler, which has no molecules.

## Library error factors

`[simulation.reads] library_error_sigma` (default 0, off) gives libraries different error susceptibilities, as DNA
damage, PCR cycles and run quality differ between real libraries. Each library draws one factor
f = exp(sigma Z - sigma^2 / 2) (lognormal, mean one; stream `simulation.library_error`, keyed by the generating seed and
the sample identifier, so it is the same on every contig and at any sample order). The library's reads carry
`error_rate` x f (paired-end and independent-site samplers; the long-read quality law is not scaled), and the
call-set layer converts its REF reads at error-prone sites with the site's rate x f. Only the increase is clipped: a
rate r becomes min(r f, max(r, `LIBRARY_ERROR_RATE_LIMIT`)) (0.25), so a configured rate above the limit is never
lowered. Since f has mean one, the mean rate over libraries is unchanged except where r f would pass the limit, which
lowers the mean a little at large sigma; sigma spreads the rates, so the median library is cleaner and a tail is
noisier. The truth records each library's factor (`library_error_factor`). Nothing is drawn
when sigma is 0.

Real libraries differ this way: TroMau's per-sample read error (the read model's per-sample estimates) runs from
6.7e-6 at the 10th percentile and 1.4e-5 at the median to 2.4e-4 at the 90th (q90/q50 = 18).

Implementation: [reads.py](../../haplotype_reconstruction/simulation/reads.py) (`library_error_factors`),
[artefacts.py](../../haplotype_reconstruction/simulation/artefacts.py) (`library_factor`).

## Chimeric long reads

`chimera_share` (long reads only; off by default) makes a share of reads chimeric: two molecules of the same library
joined during library preparation (ligation, or a missed adapter between two reads).
- **Physical choice:** an aligner maps each segment to its own locus, as a primary and a supplementary alignment, and
  soft-clips a segment too short to place. A chimeric read of drawn length L is split at a uniform junction:
  - one piece stays at the read's locus;
  - the partner piece goes to a uniform locus, drawn as the sampler draws reads, with its own homologue, so coverage
    is conserved;
  - the shorter piece is lost below `chimera_min_segment_bp` (500 bp by default). At L = 3 kb that happens to about
    a third of chimeric reads.
- **Mismapping:** with `chimera_mismap_share`, the shorter piece is instead aligned contiguously after the longer
  piece, at that piece's locus and depth envelope (a local misalignment). It then reports alleles unrelated to the
  local genotype: its source homologue's allele, its own or its donor's, at a uniform other site.
- **Only one sample's molecules are joined.** A join of two samples' molecules (a barcode chimera) would leave a
  foreign segment at its own locus, which the uniform foreign-read mode already represents.
- Correctly split chimeras change no allele at any locus. Their effect is shorter effective reads and lost short
  pieces.

## Realistic profile

`[simulation.reads] preset = "realistic"` (short reads) and `"realistic_long_reads"` (long reads) make a simulated
cross look, to the pipeline, like the real TroMau cross (116 samples; bcftools mpileup joint calls, mapping quality at
least 20, on fAstCal1.2; mean depth about 11). Until 4.21 every release-matrix simulation ran without site or sample
QC (`[quality]` auto is off for simulations) and without artefacts, so the run's estimated outlier rate stayed at
5e-5 to 6e-4 where TroMau's is 0.0048 (observation pilot) and 0.0055 (painting); 4.21's phase-set fragmentation (D3),
which needs a rate of that size, could not appear in any simulation. Under both presets the painting rate is
TroMau's, and D3 appears under RC3 and not under 4.21.0: on the f3-like design 619,915 gauge-sample phase sets
(311 times the clean control) against 2,948 (1.4 times); on the famhs-like long-read design 127,455 (75 times)
against 2,619 (1.5 times). Without missed calls the short-read rate is 0.0017 and D3 does not appear (2,626 and 2,432
gauge sets, 1.3 and 1.2 times clean). A read preset is merged as a whole TOML mapping
under every explicit key, so these presets also switch on the call-set artefacts, the structure model and site/sample
QC. An explicit `[quality] site_filter = "auto"` in a configuration turns QC back off; realistic validation designs
leave `[quality]` unset.

| mechanism | settings | physical cause |
|---|---|---|
| site and sample QC | `[quality] site_filter = "on"`, `sample_filter = "on"` | real runs filter; what survives QC is what the pipeline reads |
| site-specific low-rate errors | `[simulation.artefacts]` `systematic_sites` 0.06, each REF read read as ALT with probability Uniform(0.03, 0.15) at every cell of the site; the per-cell balance off | context errors and low-level mismapping shared by samples |
| missed variant calls | `hard_sites` 0.002, `stretch_cover` 0.006, `miss` 0.0024 (0.2 times the layer's defaults; other miss settings at their defaults); a missed cell is reported as a reference block, the only rewritten cells | a caller reporting a variant cell as reference |
| clean reads | `error_rate` 2e-5 | TroMau's fitted read error is 1.4e-5 |
| read-level mapping bias | `reference_bias` 0.02, power 2 | reads with several mismatches fail to align ([OU28](#ou28)) |
| shared low-mappability depth loss | `site_loss_fraction` 0.30 (1 kb), `site_loss_fine_fraction` 0.05 (300 bp), `site_depth_cv` 0.05 | repeats and filtered windows; site QC (r1) removes most of them |
| depth spread | `depth_cv` 0.3; `region_depth_cv` 0.08 (4 kb) | library sizes of TroMau's bulk; mild coverage waves |
| structure | `[simulation.structure] enabled` (cichlid rates) | deletions, duplications, inversions, reference misorder |
| collapsed paralogs | `collapsed_fraction` 0.055, `collapsed_mean_bp` 5000, `collapsed_psv_share` 0.02, `collapsed_carrier_frequency` 0.8 (inherited copy number) | segmental duplications the reference collapsed: raised depth and pseudo-heterozygous sites in every sample ([Collapsed paralogs](#collapsed-paralogs)) |

**Stray reads are kept; missed calls give the outlier rates.** A true reference-homozygous cell keeps its own counts,
stray ALT reads included (TroMau's AD keeps every high-quality base; there is no minimum ALT fraction for a reference
call), and only missed variant cells are rewritten ([Call-set artefacts](#call-set-artefacts)). Without missed calls
no setting of the site-specific layer reaches TroMau's outlier rates together with its kept-site statistics; with
them the rates are reached and the kept-site statistics do not move (f3-like design, 11x):

| systematic sites, ALT per REF read; missed calls | painting rate | pilot | read error | r3 | kept excess-het |
|---|---:|---:|---:|---:|---:|
| 0.12, 0.01-0.06; none | 0.0010 | 0.0005 | 8.8e-5 | 0.061 | 2.4e-4 |
| 0.06, 0.03-0.15; none | 0.0017 | 0.0007 | 5.5e-5 | 0.031 | 2.9e-4 |
| 0.03, 0.10-0.30; none | 0.0055 | 0.0010 | 4.6e-5 | 0.015 | 2.2e-3 |
| 0.06, 0.03-0.15; 0.15 x defaults | 0.0055 | 0.0033 | 5.6e-5 | 0.031 | 3.0e-4 |
| **0.06, 0.03-0.15; 0.2 x defaults (preset)** | **0.0055** | **0.0045** | **5.6e-5** | **0.031** | **3.0e-4** |
| 0.06, 0.03-0.15; 0.25 x defaults | 0.010 | 0.0055 | 5.6e-5 | 0.031 | 3.3e-4 |
| 0.06, 0.03-0.15; 1 x defaults | 0.03 | 0.022 | 5.4e-5 | 0.031 | 3.3e-4 |
| TroMau (tolerance) | 0.0055 (0.0035-0.009) | 0.0048 (0.003-0.008) | 1.4e-5 (3e-6-1e-4) | 0.074 (0.03-0.15) | 4e-4 (0-2e-3) |

- Without missed calls, single stray reads at homozygous cells are explained as read error, and the painting rate
  reaches TroMau's only when few sites carry a high per-read ALT share, which then fail r3 and the kept excess-het
  share.
- The pilot rises linearly with the miss rate (about 0.0007 + 1.8 x `miss` at these proportions); the layer's
  defaults (the long-read cohort's rates) give 4.6 times TroMau's pilot.
- What missed calls do not reproduce: TroMau's per-sample rate falls with depth (Spearman -0.67); with missed calls
  it rises with depth (+0.50 on this design; -0.23 without them), since a heterozygote with no ALT read is more
  conclusive at higher depth. TroMau's outliers are read-level; missed calls reach their level, not their kind.
- PCR duplicates left in the data (`duplicate_rate` 0.05-0.20) do not close the gap without missed calls: painting
  0.001-0.003, pilot 0.0006-0.0015, and at 0.20 the kept heterozygote balance and missingness leave their
  tolerances. TroMau's pileup skipped duplicate-flagged reads in any case (its bcftools mpileup filter flags include
  DUP).

The long-read preset runs on the same layer: 6% of sites at 3-15% ALT per REF read and missed calls at 0.12 times
the defaults give painting 0.0055 and pilot 0.0048 on a famhs-like design (without missed calls 0.0055 and 0.0039).

**What the mechanisms do, and what they do not.** At the sites TroMau's QC keeps, depth is near Poisson, the
heterozygote balance is binomial (0.492) and the fitted read error is 1.4e-5, yet the outlier rate is 0.0055 and each
sample's rate falls with its depth (Spearman -0.67; the ~50x samples 0.001, the 8-9x samples 0.012-0.03). A
discordance seen at 8x but not at 50x is a read-level one at clean-looking sites. In the presets:
- Site-specific low-rate errors set the kept-site statistics (r3, read error, kept excess-het). Without erased reads
  they give only about a third of TroMau's painting rate at kept-site statistics inside tolerance, and their rate
  falls with depth (Spearman -0.23 at depth CV 0.3, -0.35 at 0.6).
- Missed variant calls supply the rest of the outlier level, and their rate rises with depth (+0.50 at depth CV 0.3,
  +0.63 at TroMau's 0.6): a heterozygote with no ALT read is more conclusive at higher depth.
- Library error factors (`library_error_sigma`, off in the presets) reproduce TroMau's per-sample spread but not its
  level ([Library error factors](#library-error-factors)).
- So the presets match TroMau's outlier level, not its kind. Analyses that depend on how outlier rates vary across
  samples or with depth should not rely on these presets.
- Structure alone moves the rate only to 1e-4 after QC.

**Calibration (f3-like design, 11x; frozen targets; `lab422/sim`).** Final short-read preset:
- Painting rate 0.0055 and pilot 0.0045 (TroMau 0.0055 / 0.0048).
- Site-QC kept share 0.27 (0.36): the template design's informativeness rule r5 removes about half the sites even
  without artefacts.
- r3 0.031 (0.074).
- Kept heterozygote ALT share 0.488 (0.492), with binomial variance ratio 0.97 (0.98).
- Pre-QC heterozygote ALT share 0.449 (0.448).
- Read-model error 5.6e-5 (1.4e-5).
- Zero-depth cells: 6.7% of input cells (7.0%) and 6.1e-4 of kept cells (1.4e-4).
- Kept excess-heterozygous sites 3.0e-4 (4e-4).
- The per-sample rate spread q90/q50 is 1.34 (2.28; tolerance from 1.5), at noise level (1.26-1.46 across the
  arms); the per-sample rate rises with depth (above).

TroMau's pre-QC deficit is broad (median
heterozygote ALT share about 0.455 on all 22 contigs); collapsed paralogs explain part of it (biased sites that QC
then removes), and its non-reference founder lineage, which a one-population design does not have, is studied in
[A TroMau-like cross](#a-tromau-like-cross). The long-read
preset reaches painting 0.0055 and pilot 0.0048 on a famhs-like 5x design with its own sampler.

**Cost.** The mate mapping loss adds one uniform per mate and prefix-sum lookups of the mate's ALT count. Measured single-process on one 32 Mb template contig at 11x (`read_rows` timed per sample): 0.33 s
per sample clean, 0.40 s with the realistic depth law, 0.50 s with the mapping bias as well; samples are simulated in
parallel blocks, so a 112-sample contig takes seconds of wall time, against tens of minutes for the pipeline.

Implementation: [reads.py](../../haplotype_reconstruction/simulation/reads.py), presets
`interface/settings/presets/realistic_reads.toml` and `realistic_long_reads.toml`. Evaluation scores phase on the
QC-kept sites and reports `site_qc_removed_sites` ([evaluation](evaluation.md)).

## A TroMau-like cross

`configs/simulation_tromau_like.toml` (pedigree `configs/tromau_like.tsv`) simulates a hybrid cross shaped like TroMau:
three founders drawn from the reference genome's own population and one male from a lineage that split from it 30,000
generations ago (both populations of effective size 10,000), crossed as F1 = each female x the male and F2 = F1 x F1
half-sib families. 116 samples are sequenced; the 50 F2 of five families have unsequenced parents. The coalescent source
gives the male's haplotypes ALT alleles at 53% of sites against 19% for a reference-population haplotype, and
mismatch-dense stretches (three or more ALT alleles in 150 bp) nearly four times as often (6.0% against 1.6% of
150 bp windows).

**Lineage-dependent mapping loss.** The read-level mapping loss ([OU28](#ou28)) already depends on the lineage: a mate
from the diverged haplotype carries more ALT alleles, so it is lost more often. Under the realistic preset's law (0.02
k^2) the lineage design lowers the pre-QC heterozygote ALT share only from 0.478 (one-population design) to 0.473
(TroMau 0.448). A stronger law of the same power lowers it over every site: at 0.05 k^2 the share is 0.457, but the
QC-kept share falls with it (0.467 against TroMau's 0.492), because the loss widens the depth distribution and the
depth rule r1 stops removing the affected sites (at 0.1 k^2 r1 removes 5% of sites against TroMau's 51%). A
steeper law concentrates the loss on mismatch-dense reads, which then lose depth in a few places that site QC removes,
as in TroMau. The configuration uses 0.003 k^5 (0.3% of mates lost at one ALT allele, 9% at two, 52% at three, 95% at
four), calibrated on a 4 x 12 Mb version of the design (same site density) against the frozen TroMau targets:

| statistic | TroMau | preset law (0.02 k^2) | 0.05 k^2 | 0.003 k^5 (configuration) |
|---|---:|---:|---:|---:|
| pre-QC heterozygote ALT share | 0.448 | 0.473 | 0.457 | 0.467 |
| QC-kept heterozygote ALT share | 0.492 | 0.480 | 0.467 | 0.482 |
| site QC r1 / r2 removed share | 0.51 / 0.36 | 0.41 / 0.10 | 0.44 / 0.18 | 0.54 / 0.29 |
| sites with depth above 1.5x the median | 0.038 | 0.001 | 0.003 | 0.038 |
| kept zero-depth cells | 1.4e-4 | 7.4e-4 | 1.1e-3 | 7.6e-4 |

What it does not reproduce: the kept heterozygote share stays below TroMau's in every calibration arm (the drop from
pre-QC to kept is 0.010-0.016 against TroMau's 0.044), so some of TroMau's pre-QC deficit lies at sites with causes
this model does not have.

**D3 on this design** (8 x 16 Mb, 4.21's phase-set fragmentation; `lab422/sim`, BUILD.md sections 10 and 19). With
the realistic preset as it stands the run's painting outlier rate is 0.0055, TroMau's. The 50 samples without an
inferred parent or child (exactly the 50 F2 with unsequenced parents) are the gauge samples. Their phase sets:

| | clean reads (painting rate 5.5e-5) | this configuration (0.0055) |
|---|---:|---:|
| RC3 (before the F1b fix) | 1,159 | 452,130 (390x) |
| 4.21.0 | 1,167 | 2,804 (2.4x) |

On TroMau, D3 took the phase sets from 40,916 to 3.1 million under RC3. Both trees find 120 of 124 true
parent-child edges with no false edge, and genotype error on the QC-kept sites is 1.54e-3. Without the preset's
missed calls the painting rate is 0.001 and D3 does not appear (2,555 and 2,512 gauge sets; 108 of 124 edges).

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
