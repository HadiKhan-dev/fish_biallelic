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
when its mask is empty. Emit REF/ALT counts after dropout. Because no sample's draws depend on another's, blocks of
samples are simulated in parallel with the same counts at any worker count.

Implementation: [simulation/reads.py](../../haplotype_reconstruction/simulation/reads.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

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

Implementation: [artefacts.py](../../haplotype_reconstruction/simulation/artefacts.py). Parameters and units are in the [settings reference](../reference/settings.md).
