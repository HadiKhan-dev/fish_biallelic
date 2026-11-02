# S1. Read evidence, quality control and relatedness

## Input admission and the sample axis

A record is admitted when it has one ALT allele and both alleles are single nucleotides (A, C, G, T). All admitted records sharing a position are removed before AD is decoded. Missing AD gives zero reads. Input FILTER is not a quality decision (EC01). The sample axis is sorted by name, with the same permutation applied to every per-sample array, and outputs are restored to input order (EC03).

## Read-model fitting

*Folds (EC17).* Per-sample (depth, ALT) histograms of cells with $1\le d\le$ `maximum_fit_depth` = 48 are kept separately for training cells (fold $\lfloor x_j/w_{\mathrm f}\rfloor\bmod4\in\{0,1\}$, $w_{\mathrm f}$ = `fold_window_bp` = 1 Mb) and held-out cells.

*Shared binomial (EC18).* EM starts from $(e,q)\in$ `em_starts` = {(0.02, 0.5), (0.01, 0.35), (0.08, 0.65)} with genotype weights 1/3; pooled M-steps set $e$ and $q$ to expected wrong-allele (heterozygote ALT) reads over expected homozygote (heterozygote) depth, requiring $0<e<q<1-e$.

EM stops when the gain is at most `em_tolerance_per_observation` = $10^{-10}$ per observation. A start fails at `em_maximum_iterations` = 2,000, or on a likelihood decrease of more than $10^{-7}$ per observation. Each endpoint is then Newton-polished to stationarity in at most 40 box-constrained steps.

Endpoints are equivalent within relative distances of $10^{-13}$ (shared logits) and $10^{-8}$ (nuisance logits). The representative endpoint has the largest training log-likelihood; ties go to the lexicographically smallest parameters. A distinct endpoint within 1 nat of the best makes the fit unresolved. A failed polish fails its start; an unresolved fit, or failure of every start, gives the fixed model.

*Pooled sample binomial (EC19).* Each sample is first fitted without a prior, by L-BFGS-B [@byrd1995] plus a Newton polish. The variances $v_i$ of its two logits are diagonal entries of the inverse observed Hessian, so its genotype mixture is profiled out. They are used only if the smallest Hessian eigenvalue exceeds $10^{-8}$. For each logit $y$, $(\mu,\tau)$ minimise

$$
\tfrac12\sum_i\Big[\log(v_i+\tau)+\frac{(y_i-\hat\mu(\tau))^2}{v_i+\tau}\Big],\qquad 0\le\tau\le\max\{1,4\operatorname{var}(y)\},
$$

where $\hat\mu(\tau)$ is the precision-weighted mean. The minimiser is refined to a root of the derivative. $\tau=0$ is kept when its profile value lies within a relative $10^{-10}$ of the optimum.

With fewer than three finite variances, $\mu$ takes the shared value and $\tau=0$. Samples are refitted under $\mathcal N(\mu,\tau)$; a $\tau=0$ effect is fixed at $\mu$, and a sample without training reads takes the prior mode.

*Joint beta-binomial (EC20).* The fit starts from the pooled effects with $\kappa$ = `initial_heterozygote_concentration` = 40, under the pooled prior. L-BFGS-B is followed by a Newton polish on the arrowhead Hessian: one 4×4 block per sample, coupled to $\log\kappa$. The bounds are $\operatorname{logit}e\in[-16,-10^{-8}]$, $\operatorname{logit}b\in[-12,12]$, $\log\kappa\in[-8,16]$ and mixing logits in $[-25,25]$.

*Selection (EC21–EC22).* Stretches without admitted cells are dropped. Complex candidates with non-finite held-out totals are skipped, and a failed complex fit never discards a valid simpler one. Acceptance needs $n_h\ge$ `selection_minimum_blocks` = 4, $\Delta>0$ and $z\ge$ `selection_z` = 2 (inclusive), and a rejection does not stop later comparisons. The fallback model is binomial, with `fixed_error` = 0.02 and balance 1/2.

*Application (EC16, EC23).* Emissions are exact integer-depth rising products, exact in the binomial limit $1/\kappa=0$. The `lookup_depth` = 64 table changes cost only. Validation requires exactly 1/3 at zero depth and finite, nonnegative values normalised within $10^{-12}$ elsewhere.

## Site filter

*Order (EC11).* All sites are kept, with the reason recorded, when the filter is off (the default for simulations) or when there are fewer than `minimum_calibration_sites` = 1,000 SNPs. Otherwise r1, r2, r3, r4, an initial r5, the library check and the r5 recount run in that order. Removal is the logical OR of r1–r5, or of r1–r4 when `drop_uninformative_sites` = false; the default is true.

*Classification (EC04).* The classification uses `classification_error` = 0.01 and `classification_heterozygote_alt` = 0.5. Frequencies start at 1/3 with a floor of $10^{-6}$. EM runs for at most 60 iterations, stopping from the fifth on once the gain falls below $10^{-7}$. $W_j$, $D^{\mathrm{hom}}_j$, $A_j$ and $H_j$ are moments from the last E-step. The leakage terms use the final frequencies:

$$
\lambda_j=\sum_{i:d_{ij}>0}\Big[f_{j1}\textstyle\sum_{a}\mathrm{Bin}(a;d_{ij},\tfrac12)\,w(a,d_{ij})+f_{j0}w(0,d_{ij})+f_{j2}w(d_{ij},d_{ij})\Big],
$$
$$
\gamma_j=\sum_{i:d_{ij}>0}d_{ij}\Big[f_{j0}\big(w(1,d_{ij})-w(0,d_{ij})\big)+f_{j2}\big(w(d_{ij}-1,d_{ij})-w(d_{ij},d_{ij})\big)\Big],
$$

where $w(a,d)$ is the posterior-weighted wrong-read count of a cell.

*Clean core (EC05).* The histogram covers the 0.5th–99.5th percentiles with bin width $2\,\mathrm{IQR}\,n^{-1/3}$ [@freedman1981]. Integer data use rounded widths of at least 1; continuous data use 50–2,000 bins. A moving average over `histogram_smoothing_bins` = 5 bins smooths the counts. The mode is the first maximal bin, a reversible convention; tied modes are counted. Half-maximum distances divided by $\sqrt{2\ln2}$ give the two standard deviations.

The local posterior is $\min(1,\hat f_0/\hat f)$. Here $\hat f_0$ is the side-specific Gaussian at the peak density, and $\hat f$ is the smoothed density normalised by all finite observations, including those outside the histogram range. The window is the contiguous run of bins around the mode with posterior at least `core_posterior_level` = 0.5. It is unbounded where it reaches a histogram edge.

*r1, r2 (EC06–EC07).* r1 rejects only depths strictly beyond the window. r2 sums depths over the r1-passing sites before dividing, and fits its core to $\log X_j$ at the r1-passing sites with $X_j>0$. Without such sites, r2 is not applied.

*r3 (EC08).* The domain is sites passing r1–r2 with $D^{\mathrm{hom}}_j\ge1$ and $\gamma_j>10^{-9}D^{\mathrm{hom}}_j$. The fit starts from clean weight 0.6, $\varepsilon_0=\mathrm{clip}(\tfrac12\operatorname{median}[\max(W_j-\lambda_j,0)/\gamma_j],10^{-5},0.1)$, $\varepsilon_1=10\varepsilon_0$ and $\phi=1$. Each iteration updates three blocks:

- the weight, held within $[10^{-6},1-10^{-6}]$;
- $\varepsilon_0\in[10^{-12},0.5]$, by safeguarded Newton steps on its concave objective;
- $(\log\varepsilon_1,\log\phi)$, by projected Newton ascent within $\varepsilon_1\le10$ and $\phi\in[10^{-3},10^4]$.

It stops when the summed relative change falls below $10^{-5}$, or after 300 iterations. The rule applies if $\varepsilon_1\ge$ `minimum_artefact_separation` $\times\,\varepsilon_0=3\varepsilon_0$. It removes sites whose clean posterior is below `clean_posterior_level` = 0.5.

*r4 (EC09).* The domain is sites passing r1–r3 with $H_j\ge1$. A component has mean $m$ and inverse concentration $t$, with latent SD $\sqrt{m(1-m)t/(1+t)}$; $t=0$ is binomial. Five fixed starts around the pooled $\sum A_j/\sum H_j$ are each fitted by L-BFGS-B, with at most three restarts and eight Newton polish steps. A start is certified when its projected gradient is at most $10^{-8}$; the best start has the lowest objective among certified starts (among all, if none is), ties by parameters, then start. The clean component has weight $\ge1/2$.

With scale $\sigma_{\mathrm{sep}}=\max\{\text{clean latent SD},\sqrt{m_0(1-m_0)/\operatorname{median}H}\}$, the rule applies when the artefact latent SD, or $|m_1-m_0|$, is at least $3\sigma_{\mathrm{sep}}$. It abstains when $|m_0-\tfrac12|-|m_1-\tfrac12|\ge3\sigma_{\mathrm{sep}}$, or when no start is certified. This contig-level gate can miss well-resolved high-depth sites.

*r5 (EC10).* r5 rejects a site with fewer than `minimum_expected_carriers` = 2 expected carriers (a strict inequality). After library flags, it is recomputed on unflagged samples only. r1–r4 are not recalibrated.

## Library quality check (EC12–EC15, EC25)

The check runs only where the site filter applies (by default, on for VCF input), on the sites passing r1–r4.

*Eligible cells.* The cohort-minor allele is ALT when cohort ALT reads do not exceed REF reads. A library carries it at a cell with $\tilde a$ minor reads of depth $d$ when $\Pr(Y\ge\tilde a)\le\eta/(N-1)$, where $Y\sim\mathrm{Bin}(d,\varepsilon_0)$ and $\eta$ = `carrier_tail_level` = 0.05. Here $\varepsilon_0$ is r3's clean rate, or the classification error when r3 did not fit. A cell is eligible when the library is observed there and no other library carries the minor allele; cells with $\tilde a=0$ count.

*Weights and partitions.* Because nearby cells share reads, each eligible cell is weighted by the reciprocal of the library's eligible-cell count in its `fragment_bp` = 1,000 bp window. Sites are cut into `artefact_regions` = 16 regions of equal site count, and each library's cells into `heterozygote_depth_classes` = 4 depth-quantile classes.

*Mixture components.* The null mixture has, per depth class:

- binomial error components on a log grid from $0.1\varepsilon_0$ to $\alpha_{\min}$, with ratio 1.6;
- beta-binomial own-heterozygote components in both orientations, with means at r4's clean balance (or at the ends and midpoint of r4's two means when its labelling is unresolved) and latent dispersions {0, 0.0001, 0.0004, 0.0016, 0.0064, 0.0256, 0.04};
- per-region artefact shares, under which minor counts are uniform on $0..d$.

The alternative adds $c_{\mathrm f}\,\mathrm{Bin}(\tilde a;d,\alpha)$, with share $c_{\mathrm f}$ common to depth classes and regions. $\alpha$ takes 24 log-spaced values in $(\alpha_{\min},\alpha_{\max}]$, where $\alpha_{\min}$ = `minimum_foreign_fraction`/2 = 0.005 and $\alpha_{\max}$ = `heterozygote_floor_ratio` $\times\,h_{\mathrm{lo}}/2$, with $h_{\mathrm{lo}}$ the lower end of the own-heterozygote balance interval. The mixtures are fitted by EM with at most 400 iterations and tolerance $10^{-10}$. Alternatives are warm-started in ascending $\alpha$, and the null is re-polished from the best alternative's weights.

*Flagging.* A library is flagged when $\Lambda=\max_\alpha2[\ell^{(1)}(\alpha)-\ell^{(0)}]\ge u_N$, the difference of maximised alternative and null log-likelihoods at the first maximum. The threshold $u_N$ satisfies $\Pr(\chi^2_1\ge u_N)=2\eta/(24N)$. This is a Bonferroni level under the boundary law $(\chi^2_0+\chi^2_1)/2$ [@self1987]. It is nominal only for a regular, correctly specified likelihood and is documented as a working threshold.

*Inapplicability.* The check is recorded as inapplicable when any of the following holds:

- fewer than `minimum_samples` = 20 samples have observed cells;
- no cells are eligible;
- $2\varepsilon_0\ge \alpha_{\min}$;
- $\alpha_{\max}\le1.25\,\alpha_{\min}$.

*Effect of a flag.* A flag hides the library's cells only in the founder-construction view and in r5.

*Inherited segments (EC15).* A diagnostic report, never a flag or parent call, groups near-private cells shared with one partner (gaps below `segment_gap_bp` = 1 Mb; at least `segment_minimum_events` = 3 events).

*Discovery sites and blocks (EC24).* Discovery sites have an integer total depth of at least `minimum_total_depth` = 5. Blocks are runs of `block_sites` = 200 consecutive retained sites, the last possibly shorter, and ignore spacing and the discovery mask.

## Genetic map (EC27)

A map needs at least two knots, with finite, nonnegative, strictly increasing positions and finite, nondecreasing cM values. The model coordinate interpolates between knots and extends beyond them at `rate_cm_per_mb` (default 5.0). Without a map, it is that rate times position. The measured coordinate equals the model coordinate if `rate_is_measured` is true. Otherwise (the default), measured coordinates clip positions to the terminal knots, and a chromosome without a map has zero measured length. The Haldane fraction is computed as $-\tfrac12\operatorname{expm1}(-2\delta)$ (ledger entry L1); each consuming model sets its own probability bounds. The TroMau configuration supplies no map file and uses the 5 cM/Mb fallback with `rate_is_measured` false.

## Tie rules and shared emissions (EC30–EC33, EC35)

Shared tie quanta are powers of two, set per decision kind from the decision's scale and the contrast's measured motion under $10^{-6}$ relative genotype-likelihood noise. Classes link consecutive sorted exact values. A class wider than `WIDE_CLASS_QUANTA` = 4 quanta is ordered by value inside. Improvement is strict; content keys are exact discrete data. Reversal flips only arbitrary conventions, such as the histogram-mode tie, and enters the derivation. The read-model representative and r4's best start use lexicographic parameter order. Relatedness uses `gain_tolerance` = $10^{-6}$ nats with evidence keys.

Founder-pair emissions (EC35; assembly and founder refinement) score a sample's normalised triple $P$ under an ordered founder pair as $P(g)$ for called dosage $g$, $\tfrac12[P(0)+P(1)]$ or $\tfrac12[P(1)+P(2)]$ for one called allele (0 or 1), $\tfrac14P(0)+\tfrac12P(1)+\tfrac14P(2)$ for two unknown alleles of different rows, and $\tfrac12[P(0)+P(2)]$ for two copies of one unknown allele (a per-observation approximation). The robust score of a predictive likelihood $\rho$ is $\max(\log(0.99\rho+0.01/3),-2)-\log\tfrac13$; uniform or zero-sum triples contribute nothing.

## Relatedness (RL01–RL19)

*Inputs (RL01–RL03, RL05).* Contigs are concatenated in configuration order, fixing ranks and reduction order. Allele frequencies take `iterations` = 6 EM steps from `initial` = 0.5, with update $p\leftarrow N^{-1}\sum_i[\tfrac12\omega_{i1}+\omega_{i2}]$ ($\omega_{ig}$ posterior genotype probabilities) clipped at `clip` = $10^{-4}$. Every sample counts in $N$, and unobserved samples contribute their prior weights. Site gates are `minor_allele_frequency` = 0.05 and `minimum_observed` = 0.5, both inclusive. Thinning keeps every `family_site_step` = 20th passing site, starting from the first. Log terms are floored at $10^{-300}$.

*Keys (RL07).* A sample's evidence rank is the dense lexicographic rank of its full child-term vector, ordered by site and then by genotype pair. Equal vectors are exchangeable. A family key lists member ranks, then parent ranks (latent = −1), each in descending order. A pair key is the two family keys, sorted. Ties within `gain_tolerance` go to the larger key.

*Agglomeration (RL09).* Pairs whose full-sib gain is strictly positive and strictly above their half-sib gain are merged best-first, with affected entries rescored, until none remain.

*Parent assignment (RL10).* Each family's first free slot is offered the best sequenced non-member (excluding members and the other parent). In families of two or more, each member $i$ is also offered: $i$ moves to its best other placement and fills the slot. Offers are ranked on the sweep's starting partition by gain, parent rank and family key, and applied only if:

- its rescored gain exceeds $\log N$;
- neither its family nor its parent has been used in this sweep;
- its slot is still free;
- the parent graph stays acyclic.

No second-best search follows a failure.

*Moves (RL11–RL12).* A sample may not join a family whose sequenced parent is itself or its descendant. The best-improvement search applies, up to $N$ times per sweep, the rescored largest move gain above $10^{-6}$ (ties by evidence rank, then target key, the singleton target included). The content-order search visits each sample once, by decreasing rank, moving it when its best gain exceeds $10^{-6}$.

*Merges and sweeps (RL13–RL14).* Two families are compatible when their sequenced parents number at most two, none being a member of either family. Latent pairs must also beat their half-sib link. Offers above $10^{-6}$ are ranked by gain, then pair key. The first acyclic merge is applied, and a cyclic trial is undone without reordering. A sweep consists of parent assignment, moves and one merge, up to `max_sweeps` = 30. Convergence and the margin between the searches are recorded.

*Screens (RL15).* Float32 screens bound their error by $0.1+10^{-6}M$ nats plus $2(M+16)2^{-53}$ times the summed log-likelihood magnitudes. Comparisons the bound leaves undecided are evaluated exactly. Caches hold only gains of unchanged families; screens never change decisions.

*Support (RL16–RL17).* Each chromosome's measured length between family sites is cut into $\max(1,\lfloor\text{length}/1\,\mathrm{M}\rfloor)$ equal units (`support_unit_morgans` = 1). Cuts map to the first site at or above them; empty units are dropped. A member wins a unit when its stay value, $\ell_F-\ell_{F\setminus i}$ (or $\ell_F$ if alone), beats both alternatives by more than $10^{-6}$. The alternatives are its best join gain $\ell_{G\cup i}-\ell_G$, excluding its own family and families it parents, and being alone with latent parents. A latent one-member family therefore scores 0. Parent support compares $\ell_F$ with the named parent and with a latent parent; it is missing for latent slots.

*Links and numbering (RL18–RL19).* After the kept fit, even one stopped at the sweep cap, every pair of fully latent families with a link gain above $\log N$ is reported, with no full-sib comparison required. Families are numbered by the rank of their smallest member.

*Pairwise IBD (RL04).* The chain uses every `ibd_site_step` = 10th retained site, without the family-site gates. For two samples' triples $P,P'$, $\psi=\psi_j$ and $p=p_j$, the state emissions are

$$
o_0=\Big(\sum_g\psi(g)P(g)\Big)\Big(\sum_g\psi(g)P'(g)\Big),\qquad o_2=\sum_g\psi(g)P(g)P'(g),
$$
$$
o_1=(1-p)^3P_0P'_0+p^3P_2P'_2+p(1-p)P_1P'_1+p(1-p)^2(P_0P'_1+P_1P'_0)+p^2(1-p)(P_2P'_1+P_1P'_2).
$$

Sites not jointly observed emit 1. The chain starts uniform and stays with probability $\exp(-\theta\delta_j)$ over the model-map distance $\delta_j$ from the previous chain site, with $\theta$ = `ibd_switch_rate` = 4 per Morgan; it leaves to each other state with probability $[1-\exp(-\theta\delta_j)]/2$. Posteriors are averaged over the jointly observed sites. The IBD0 (IBD2) ratio compares the full chain with one lacking IBD0 (IBD2).
