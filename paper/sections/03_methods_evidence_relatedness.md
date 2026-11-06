## Read evidence, quality control and relatedness

### Read evidence and genotype likelihoods

**Observations.** The input is an indexed multi-sample VCF or BCF file [@danecek2011; @danecek2021] with per-sample allele depths (FORMAT/AD). A record is admitted only if it is a biallelic SNP with single-nucleotide REF and ALT alleles. When two or more admitted records share a position, all of them are removed. Missing AD values count as zero reads; input FILTER flags and called genotypes do not enter as evidence. The observation for sample $i$ at site $j$ is the read pair $(r_{ij},a_{ij})$ with depth $d_{ij}=r_{ij}+a_{ij}$, and a cell with $d_{ij}=0$ is unobserved. Samples are ordered canonically by name, and names are identifiers only: no generation, cohort, sex or parentage information enters any stage.

**Read model.** Given the genotype $g$, the ALT count of a cell follows

$$
\Pr(a_{ij}\mid d_{ij},g)\;\propto\;
\begin{cases}
e_i^{\,a_{ij}}\,(1-e_i)^{\,r_{ij}}, & g=0,\\[4pt]
\dfrac{\mathrm{B}\!\left(a_{ij}+\kappa q_i,\; r_{ij}+\kappa(1-q_i)\right)}{\mathrm{B}\!\left(\kappa q_i,\;\kappa(1-q_i)\right)}, & g=1,\\[10pt]
(1-e_i)^{\,a_{ij}}\,e_i^{\,r_{ij}}, & g=2,
\end{cases}
\qquad q_i=e_i+(1-2e_i)\,b_i ,
$$

where the binomial coefficient common to the three genotypes is omitted. Here $e_i$ is a homozygote's wrong-allele read probability, and the heterozygote ALT probability $q_i$ combines it with an allelic balance $b_i$. Heterozygote reads are overdispersed through a beta-binomial law with concentration $\kappa$, and $\kappa\to\infty$ recovers the binomial heterozygote. Homozygotes are binomial in every model.

**Calibration and model choice.** The read model is fitted per contig, from its allele depths alone, on the sites kept by the quality filter below. Three nested candidates are fitted:

1. a shared binomial model, with one $e$ and one $q$ for all samples and $\kappa=\infty$, fitted by expectation–maximisation [@dempster1977] from several starts;
2. a sample binomial model, with per-sample effects $\operatorname{logit}e_i$ and $\operatorname{logit}b_i$ under normal priors whose mean and between-sample variance are estimated by empirical Bayes, each sample's sampling variance coming from a diagonal Laplace approximation;
3. a sample beta-binomial model, which refits these effects jointly with one shared $\kappa$ under the same priors.

Each sample's genotype proportions are fitted as nuisance parameters. They are never returned as genotype priors, so the genotype likelihoods carry read evidence only.

Fitting and model choice use different physical folds. A site at position $x_j$ (bp) falls in fold $\lfloor x_j/10^{6}\rfloor \bmod 4$ (1-Mb windows). Folds 0 and 1 train, folds 2 and 3 are held out, and only cells with $1\le d_{ij}\le 48$ enter fitting and scoring. Held-out log predictive likelihoods are summed over each held-out stretch, which is the held-out half of one four-window cycle. Candidates are compared in order of complexity. Each one replaces the currently accepted model when its total held-out gain $\Delta$ is positive, at least four stretches contribute, and

$$
z=\frac{\Delta}{\sqrt{n_h\, s_\Delta^2}}\;\ge\;2,
$$

where $s_\Delta^2$ is the sample variance of the $n_h$ stretch gains. The standard error comes from whole stretches, not single sites, because sites linked through ancestry or sharing reads are not independent observations. A fixed binomial model ($e=0.02$, $q=1/2$) is used instead when calibration is disabled, a fold has no admitted cells, the shared model cannot be fitted or resolved to one optimum, or its held-out score is not finite.

**Genotype likelihoods.** The selected model is applied to every cell of the contig, at any depth and without a genotype prior. Its three values are normalised,

$$
P_{ij}(g)=\frac{\Pr(a_{ij}\mid d_{ij},g)}{\sum_{g'=0}^{2}\Pr(a_{ij}\mid d_{ij},g')},
$$

and an unobserved cell is exactly neutral, with $P_{ij}(g)=1/3$. The reads of every sample, including libraries flagged by the check below, enter calibration and the genotype likelihoods.

The evidence representation is exact: per sample and contig, repeated integer read-count pairs
share a codebook entry containing that pair and its original float64 likelihood triple. Codes
occupy 1, 2 or 4 bytes per site and sample, in addition to the codebook; samples remain dense
when coding would not reduce their joint likelihood, count and mask storage. Consumers gather
the same ordered dense values, bit for bit. This changes storage rather than the read model or
observation mask (Supplementary Information, section S6).

### Site and library quality control

The site filter calibrates itself on each contig's own depth and allele-count distributions, without sample labels, pedigree, benchmark or simulated truth. It is on by default for VCF input and applies to contigs with at least 1,000 admitted SNPs; on smaller contigs every site is kept.

The rules use a per-site genotype classification: an expectation–maximisation fit over three free genotype frequencies $f_{j0},f_{j1},f_{j2}$ (no Hardy–Weinberg constraint) under a fixed binomial classification model ($e=0.01$, heterozygote ALT fraction 1/2). It yields expected wrong-allele reads $W_j$ and depth $D^{\mathrm{hom}}_j$ in homozygotes, and expected ALT reads $A_j$ and depth $H_j$ in heterozygotes. They also use a "clean core": the mode of a smoothed histogram, with separate Gaussian-equivalent left and right widths from its half-maximum crossings. Its window is the region around the mode where the core's local posterior probability is at least 0.5, an empirical-null construction [@efron2004].

Five rules can remove a site:

- **r1** removes a site whose total depth $D_j=\sum_i d_{ij}$ lies strictly outside the clean-core window of total depth, in either tail. The window is calibrated on the contig, not a fixed global depth threshold.
- **r2** removes sites above the clean-core window of $\log X_j$, where the depth-heterogeneity index is $X_j=(N-1)^{-1}\sum_i (d_{ij}-E_{ij})^2/E_{ij}$. The expected depth is $E_{ij}=D_j\bar d_i/\sum_{i'}\bar d_{i'}$, with $\bar d_i$ sample $i$'s mean depth over sites passing r1.
- **r3** fits a two-class mixture to $W_j$. Clean sites follow $\mathrm{Poisson}(\lambda_j+\gamma_j\varepsilon_0)$. Artefact sites follow a negative binomial with mean $\lambda_j+\gamma_j\varepsilon_1$ and shape $\phi$. The term $\lambda_j+\gamma_j e$ is the first-order expectation of $W_j$ at a clean site with error rate $e$, including reads from heterozygotes and from misclassified homozygotes. The rule is applied only when $\varepsilon_1\ge 3\varepsilon_0$, and it removes sites whose posterior probability of being clean is below 0.5.
- **r4** fits a two-component beta-binomial mixture to $(A_j,H_j)$, labels the majority component clean, and removes sites whose clean posterior is below 0.5. It does so only when the minority component separates in latent spread or in mean, and abstains when the labelling is ambiguous (the majority mean lies farther from 1/2 than the minority mean) or no start of the fit reaches its stationarity certificate.
- **r5** (informativeness) removes sites with $N^{\mathrm{obs}}_j(1-\max_g f_{jg})<2$: fewer than two expected carriers of a non-major genotype among the $N^{\mathrm{obs}}_j$ samples observed at the site.

A per-library check then tests for a foreign minor-read component, using cells at which only that library carries the cohort-minor allele. On each contig, a likelihood ratio compares a null mixture of the library's own error rates, its private heterozygotes and region-specific local artefacts with an alternative that adds a binomial foreign component of read fraction $\alpha$. A library whose statistic reaches a working, Bonferroni-type threshold is flagged on that contig. A flag records a pooled likelihood preference, not a contamination diagnosis. A flagged library is treated as missing for founder construction on that contig and is left out of the recomputed r5 carrier count; every other stage still uses its reads.

Founder discovery and local selection fit alleles only at discovery sites, where the total depth across all samples is at least five reads. This is a floor on read support, not a required fraction of observed samples. The retained site axis is cut into consecutive blocks of 200 sites, regardless of physical spacing.

### Genetic coordinates

A genetic map gives cumulative centimorgans at knots with strictly increasing positions and nondecreasing map values; it is never sorted or repaired. The map is linearly interpolated between knots and extended continuously beyond them, or replaced on chromosomes without a map, at a fallback rate (5 cM/Mb by default). Model distances include this extension. Measured coordinates count only length the supplied map supports, unless the fallback rate is declared to be a measured genome-wide average. Where a stage needs a recombination fraction, a distance of $\delta$ Morgans is converted with the Haldane function $r_H(\delta)$ defined above [@haldane1919], always in the same numerically stable form.

### Relatedness

Relatedness infers full-sib families and their parents from genotype likelihoods alone; no painting, pedigree, phase or cohort label enters. It reads the evidence of every configured contig, for all samples on the filtered sites, so it starts only once all evidence exists. Founder reconstruction does not depend on it; only pedigree inference consumes it.

**Sites and frequencies.** At each site an ALT frequency $p_j$ is estimated under Hardy–Weinberg proportions by six expectation–maximisation steps from $1/2$, clipped to $[10^{-4},1-10^{-4}]$. Unobserved samples contribute their prior expectation. Family sites are those with $\min(p_j,1-p_j)\ge 0.05$ that are observed in at least half of the samples, thinned to every 20th such site. The $M$ family sites of all contigs are concatenated in configuration order. We write $\psi_j(g)$ for the Hardy–Weinberg genotype probabilities at $p_j$, and $T(g\mid g_u,g_v)$ for the Mendelian probability of offspring genotype $g$ given parental genotypes $g_u$ and $g_v$.

**Composite family likelihood.** For each sample, a child term is defined over the nine parental genotype pairs:

$$
C_{ij}(g_u,g_v)=\log\sum_{g}\tilde P_{ij}(g)\,T(g\mid g_u,g_v),\qquad \tilde P_{ij}(g)=\frac{P_{ij}(g)}{\max_{g'}P_{ij}(g')},
$$

with $C_{ij}\equiv0$ for unobserved cells. A family $F$ consists of members and two parent slots, $u$ and $v$, which carry no sex. A slot is either latent, with law $\Pi_j=\psi_j$, or filled by a sequenced sample $i'$. In that case its law is $i'$'s genotype posterior from its own reads, $\Pi_j(g)\propto\tilde P_{i'j}(g)\,\psi_j(g)$. The family log-likelihood is

$$
\ell_F=\sum_{j=1}^{M}\log\sum_{g_u,g_v}\Pi^{u}_{j}(g_u)\,\Pi^{v}_{j}(g_v)\exp\Big(\sum_{i\in F}C_{ij}(g_u,g_v)\Big),
$$

and the objective is $\mathcal{L}=\sum_F\ell_F$ over a partition of all samples into families. A sequenced parent's reads are thus counted once, in its own family; in the families it parents, its law is the genotype posterior that conditions on those reads. Families are treated as independent given these laws, so $\mathcal{L}$ is a composite likelihood [@varin2011], not a joint pedigree likelihood. Two families $F$ and $G$ with latent parents may instead share one latent parent, with half-sib link likelihood

$$
\ell^{\mathrm{hs}}_{FG}=\sum_j\log\sum_{g_u}\psi_j(g_u)\Big[\sum_{g_v}\psi_j(g_v)e^{S_{Fj}(g_u,g_v)}\Big]\Big[\sum_{g_w}\psi_j(g_w)e^{S_{Gj}(g_u,g_w)}\Big],
$$

where $S_{Fj}=\sum_{i\in F}C_{ij}$.

**Search.** The search starts from one-member families with latent parents. Agglomeration then repeatedly merges the pair of families whose full-sib merge gain is the largest, provided that gain is positive and strictly greater than the gain from linking the same pair as half-sibs. Sweeps follow, each with three steps:

1. **Parent assignment.** A free parent slot is offered its family's best sequenced non-member and each of the family's members, a member first moving to its best other placement. An assignment must raise $\mathcal{L}$ by more than $\log N$ and keep the parent graph acyclic.
2. **Single-sample moves.** A sample may move to its best admissible family or to a new family of its own.
3. **One merge.** At most one merge is made per sweep, between families with compatible sequenced parents. Two latent families merge only when the full-sib gain beats their half-sib link.

Two searches run from the same agglomerated start: a best-improvement search, which applies the single largest move over all samples, and a first-improvement search, which visits samples in a fixed content order. Both are run because the objective has many local optima and neither search reaches the better one on every design. The content-order result is kept only when its objective exceeds the other's by more than $10^{-6}$ nats. Each search stops after a sweep that changes nothing, or after 30 sweeps. Gains within $10^{-6}$ nats are tied, and a tie goes to a content key: the lexicographic rank of each sample's complete child-term vector, never its name or index. Both searches are bounded heuristics, not an exhaustive search over pedigrees.

**Support and links.** Support is counted on units of measured map length, cut from each chromosome into equal pieces of at least 1 Morgan. Fallback distance is not treated as measured support, so a chromosome without measured length forms one unit. A member's support is the fraction of units on which its own family is strictly its best placement. A sequenced parent's support is the fraction of units on which it beats a latent parent in the same slot. After the fit, pairs of entirely latent families whose half-sib link gain, $\ell^{\mathrm{hs}}_{FG}-\ell_F-\ell_G$, exceeds $\log N$ are reported as a diagnostic.

**Optional pairwise IBD.** On request, each sample pair is run through a three-state (IBD0, IBD1, IBD2) hidden Markov chain [@lander1987; @rabiner1989] on every 10th retained site, with genotype-likelihood emissions at frequencies $p_j$ [@thompson1975] and state persistence $\exp(-4\,\delta)$ over a model-map distance of $\delta$ Morgans. It reports forward–backward [@baum1970] posterior state probabilities averaged over jointly observed sites, and likelihood ratios against chains without IBD0 or IBD2. This quadratic-cost report is off by default and does not feed the family fit.
