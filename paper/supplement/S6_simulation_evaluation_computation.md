# S6. Simulation, evaluation and computation

## Simulation settings and constants

The following table lists every simulation setting with its default. Settings in `[simulation]` and `[simulation.reads]` describe the generating model only; inference keeps its own read model and map. The order of `[input] contigs` is part of the random streams, and every contig's template must hold the same even number $H$ of haplotypes.

| Setting | Default | Allowed values | Meaning |
|---|---|---|---|
| `seed` | 400 | integer or `"random"` | Seed; `"random"` takes eight OS-random bytes once at initialisation and records the realised value |
| `templates` | none (required) | directory of `<contig>.npz` | `positions` int64[$L$], `allele_probabilities` float64[$H$, $L$, 2] |
| `cohorts` | [20, 100, 200] | each $\ge2$ | Cohort sizes; cohort 1 descends from the founders |
| `depth` | 5.0 | finite, $>0$ | Mean depth $\lambda$ |
| `generating_rate_cm_per_mb` | 5.0 | — | Fallback rate $\gamma$ (5 cM/Mb $=5\times10^{-8}$ Morgans per bp) |
| `generating_map` | none | cumulative cM map | Generating map, independent of inference |
| `backcross_fraction` | 0.0 | $[0,1]$ | Per-child backcross probability in eligible cohorts |
| `backcross_start_cohort` | 3 | $\ge2$ | First cohort that may backcross |
| `observe_cohorts` | [] (all) | unique, generated cohort numbers | Cohorts eligible for sequencing |
| `observed_fraction` | 1.0 | $(0,1]$ | Sequenced fraction $\theta$ of eligible individuals |
| `pedigree` | none | TSV: sample, parent1, parent2, sex, generation, observed | Explicit pedigree; replaces `cohorts` and random matings; `backcross_fraction`, `observe_cohorts` and `observed_fraction` must keep their defaults |
| `reads.error_rate` | 0.02 | $(0,0.5)$ | $\varepsilon$ |
| `reads.depth_cv` | 0.0 | finite, $\ge0$ | $v$ |
| `reads.heterozygote_alt_probability` | 0.5 | $(0,1)$ | $b$ |
| `reads.dropout_fraction` | 0.0 | $[0,1)$ | $\phi$ |
| `reads.read_length_bp` | 0 | integer $\ge0$ | $\ell$; $\ell>0$ selects fragments |
| `reads.fragment_mean_bp` | 375.0 | $>0$ when $\ell>0$ | $\mu_{\mathrm{frag}}$ |
| `reads.fragment_sd_bp` | 150.0 | $\ge0$ when $\ell>0$ | $\sigma_{\mathrm{frag}}$ |

The next table lists the internal constants, which are not user settings.

| Constant | Value | Role |
|---|---|---|
| `MUTATION_RATE_PER_BP` | $10^{-10}$ | $\mu$, per bp per meiosis |
| `MINIMUM_FRAGMENT_BP` | 50 | Lower clip of insert length |
| `FRAGMENT_SD_LIMIT` | 6 | Upper clip $z_{\max}=\lceil\mu_{\mathrm{frag}}+6\sigma_{\mathrm{frag}}\rceil+1$ bp |
| `FRAGMENT_QUANTILES` | 100,000 | Quadrature points for the expected bases covered per fragment |
| `WINDOW_BP`, `WINDOW_SHIFT_BP` | 100,000; 50,000 | Template windows and stride |
| `TEMPLATE_HAPLOTYPES` | 6 | Linker path budget |
| `UNDECLARED_LENGTH_BP` | 300,000,000 | Contig length when the VCF header gives none |
| `SIMILARITY_CUTOFF` | 50 (%) | Junction mismatch cutoff |
| `NODE_PENALTY`, `EDGE_PENALTY` | 10; 10 | Penalties applied after each extracted path |
| `GAP_SCORE`, `EDGE_DIVISOR` | 1; 3 | Score of non-overlapping windows; edge weight = score / 3 |
| `BISECTION_STEPS` | 60 | Inversion of merged information |
| `PATH_TOLERANCE`, `CANCELLATION_TOLERANCE` | $10^{-9}$ | Path reconstruction; exact information cancellation |

The validation configurations are summarised below. The example generator illustrates software
wiring and is not an accuracy experiment.

## Validation datasets and configurations

The matrix introduced for 4.2 and rerun for 4.4 contains 39 runnable entries; rows retained only as historical, non-runnable
matrix entries are excluded. Fifteen distinct simulated datasets have truth: the thirteen design
cases and the two references named `zero_std_n80_ref` and `zero_stress_n80_ref`. Their other
variants repeat inference on the same generated data. `n80` and `stress` abbreviate those two
reference names in the manuscript. Each table row is one configured seed, not a replicated
factorial cell. `std` and `seed` repeat the same cohort design under different seeds.

**Supplementary Table S3. Simulation matrix.** $N$ is observed samples; $H$ is true founder
haplotypes, distinct from the inferred component count $K$; depth is mean read depth; $v$ is
between-sample depth CV, $\varepsilon$ homozygous read-error probability, $b$ heterozygous ALT
probability, $\phi$ dropout fraction, and BC is backcross probability from cohort 3. All inference
fallback rates are 5 cM/Mb and marked unmeasured. “Knots” denotes the explicit map described below.

**A. Cross design and observation.**

| Scenario | Seed | $N$ | Chr. | $H$ | Generated cohorts | Observed cohorts | $\theta$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| std | 9101 | 40 | 8 | 6 | 10,15,15 | all | 1.0 |
| seed | 9217 | 40 | 8 | 6 | 10,15,15 | all | 1.0 |
| n80 | 9101 | 80 | 8 | 6 | 20,30,30 | all | 1.0 |
| backcross | 6202 | 80 | 3 | 6 | 20,30,30 | all | 1.0 |
| bc8 | 6202 | 80 | 8 | 6 | 20,30,30 | all | 1.0 |
| longbc | 32011 | 80 | 3 | 6 | 20,30,30 | all | 1.0 |
| f3 | 9603 | 112 | 3 | 6 | 16,4,96 | 1,3 | 1.0 |
| longf3 | 32012 | 112 | 3 | 6 | 16,4,96 | 1,3 | 1.0 |
| f3x8 | 9608 | 112 | 8 | 6 | 16,4,96 | 1,3 | 1.0 |
| sibrich | 32001 | 112 | 8 | 6 | 16,4,96 | 1,3 | 1.0 |
| tl | 2171 | 115 | 8 | 6 | explicit: 3, 34, 96 | table flags | 1.0 |
| famhs | 32002 | 60 | 8 | 6 | explicit: 3, 12, 38, 9 | table flags | 1.0 |
| hsrand | 32003 | 66 | 8 | 6 | 6,60 | all | 1.0 |
| stress | 9102 | 64 | 8 | 6 | 20,30,30 | all | 0.8 |
| k30 | 9101 | 80 | 5 | 30 | 60,20 | all | 1.0 |

**B. Read model, backcrossing and maps.**

| Scenario | Depth | $v$ | $\varepsilon$ | $b$ | $\phi$ | Read bp | BC fraction | Generating cM/Mb | Inference map |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| std | 5.0 | 0.0 | 0.02 | 0.5 | 0.0 | 0 | 0.0 | 5.0 | fallback |
| seed | 5.0 | 0.0 | 0.02 | 0.5 | 0.0 | 0 | 0.0 | 5.0 | fallback |
| n80 | 5.0 | 0.0 | 0.02 | 0.5 | 0.0 | 0 | 0.0 | 5.0 | fallback |
| backcross | 5.0 | 0.0 | 0.02 | 0.5 | 0.0 | 0 | 0.5 | 5.0 | fallback |
| bc8 | 5.0 | 0.0 | 0.02 | 0.5 | 0.0 | 0 | 0.5 | 5.0 | fallback |
| longbc | 5.0 | 0.0 | 0.02 | 0.5 | 0.0 | 0 | 0.5 | 5.0 | knots |
| f3 | 10.5 | 0.25 | 6.7e-05 | 0.5 | 0.0 | 125 | 0.0 | 2.0 | fallback |
| longf3 | 10.5 | 0.25 | 6.7e-05 | 0.5 | 0.0 | 125 | 0.0 | 5.0 | knots |
| f3x8 | 10.5 | 0.25 | 6.7e-05 | 0.5 | 0.0 | 125 | 0.0 | 2.0 | fallback |
| sibrich | 10.5 | 0.25 | 6.7e-05 | 0.5 | 0.0 | 125 | 0.0 | 2.0 | fallback |
| tl | 5.0 | 0.0 | 0.02 | 0.5 | 0.0 | 0 | 0.0 | 5.0 | fallback |
| famhs | 5.0 | 0.25 | 0.02 | 0.5 | 0.0 | 0 | 0.0 | 5.0 | fallback |
| hsrand | 5.0 | 0.25 | 0.02 | 0.5 | 0.0 | 0 | 0.0 | 5.0 | fallback |
| stress | 5.0 | 0.6 | 0.04 | 0.6 | 0.1 | 0 | 0.5 | 5.0 | fallback |
| k30 | 5.0 | 0.0 | 0.02 | 0.5 | 0.0 | 0 | 0.0 | 5.0 | fallback |

The eight-chromosome set for `std`, `seed`, `n80`, `stress`, `bc8`, `tl`, `famhs` and `hsrand`
is chr1, chr2, chr8, chr10, chr13, chr18, chr19 and chr20. The eight-chromosome omitted-parent
set (`sibrich`, `f3x8`) replaces chr18 by chr9. All three-chromosome designs use chr2, chr9 and
chr19. The $H=30$ design uses five short template contigs m1–m5, each with 8,000 sites,
with spans of 634,611, 468,400, 586,577, 627,850 and 511,719 bp. Its template-construction seed
is 1, separate from the pedigree/read seed 9101.

The empirical-template manifest records six haplotypes per chromosome, source discovery
products and content hashes. It does not identify the original biological sampling or input VCF
beyond those products. The $H=30$ manifest records crops sourced from chr19, chr8, chr13, chr20
and chr10, respectively; these are a constructed count stress test rather than thirty documented
biological founder sequences. Template probabilities enter the simulator as described above.

For cohort designs, the listed cohort sizes exclude the unobserved template founders. The
omitted-parent designs generate cohorts (16, 4, 96), sequence only cohorts 1 and 3, and therefore
have no observable parent edges. The explicit `tl` table has 133 generated individuals (3, 34,
96 by its descriptive generation groups), with 115 observed; `famhs` has 62 (3, 12, 38, 9), with
60 observed. Those tables' observed flags replace random thinning. Sex and generation columns
are simulation truth descriptions only. Every other row uses the cohort sampler and its
configured observation fraction.

Fragment simulations use 125 bp reads with insert mean 375 bp and SD 150 bp. Independent-site
simulations set read length to zero. All rows not using fragments retain the same dormant insert
settings. The stress reference combines $v=0.6$, $b=0.6$, $\phi=0.1$, $\varepsilon=0.04$,
backcross fraction 0.5 and observation fraction 0.8; it does not isolate their effects.

All generating maps are uniform fallbacks at the rates in Table S3, with no generating map file.
`longbc` and `longf3` use an explicit inference map with first knots at position 1 and 0 cM,
and last knots (bp, cM) of (39,628,896, 198.144480) on chr2, (37,982,278, 189.911390) on chr9,
and (31,954,172, 159.770860) on chr19. Thus its interior slope is 5 cM/Mb and its knots provide
map length to the release rules, unlike an unmeasured fallback. These experiments change map
support, not just crossover rate; we use the recorded knots as a synthetic map-support input and
make no claim of empirical provenance.

**Supplementary Table S4. Phase by supported parent edges.** Rows group samples by the actual
post-guard edge product used by phase, including marginal edges. They are not groups defined by
true parents or cohorts. A zero denominator is undefined; the corresponding N50 is shown as a dash.
Spans and ratios pool chromosomes within one run only.

| Scenario | Supported parent edges | $N$ | Genotype errors / calls | Phased / correct true heterozygotes | Switches | Long switches | Released-set N50 (Mb) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| std | 0 | 10 | 8007/23167772 | 4912889/5179823 | 5 | 1 | 36.315 |
| std | 1 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| std | 2 | 30 | 8776/70806131 | 13213171/13464156 | 14 | 0 | 36.316 |
| seed | 0 | 10 | 4342/23382342 | 4652196/4856587 | 8 | 0 | 32.399 |
| seed | 1 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| seed | 2 | 30 | 6263/70831948 | 12760682/12917017 | 14 | 0 | 36.295 |
| n80 | 0 | 20 | 2787/47220303 | 10318693/10529198 | 4 | 0 | 36.310 |
| n80 | 1 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| n80 | 2 | 60 | 12983/141719342 | 27030210/27351284 | 10 | 2 | 36.316 |
| backcross | 0 | 80 | 5792/77938894 | 14208601/14358269 | 1 | 1 | 4.716 |
| backcross | 1 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| backcross | 2 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| bc8 | 0 | 20 | 2268/47244076 | 10055605/10188412 | 0 | 0 | 35.621 |
| bc8 | 1 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| bc8 | 2 | 60 | 13160/141719151 | 25499544/25882411 | 3 | 0 | 36.298 |
| longbc | 0 | 20 | 815/19480748 | 3930447/3960588 | 0 | 0 | 31.953 |
| longbc | 1 | 2 | 146/1948346 | 267288/277657 | 0 | 0 | 31.953 |
| longbc | 2 | 58 | 4312/56501244 | 9469672/9673075 | 1 | 1 | 37.599 |
| f3 | 0 | 112 | 826/109152120 | 18652911/18797686 | 21 | 0 | 17.508 |
| f3 | 1 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| f3 | 2 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| longf3 | 0 | 112 | 1781/109084847 | 20064580/20251987 | 8 | 2 | 6.103 |
| longf3 | 1 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| longf3 | 2 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| f3x8 | 0 | 112 | 3078/270232710 | 50025591/50431364 | 127 | 2 | 11.523 |
| f3x8 | 1 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| f3x8 | 2 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| sibrich | 0 | 112 | 2551/270316348 | 50976343/51517801 | 23 | 2 | 15.518 |
| sibrich | 1 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| sibrich | 2 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| tl | 0 | 61 | 11112/144119947 | 27580738/27895961 | 3 | 1 | 7.299 |
| tl | 1 | 28 | 5118/66146713 | 12242056/12588579 | 0 | 0 | 31.949 |
| tl | 2 | 26 | 3474/61422994 | 11981561/12168306 | 2 | 0 | 35.620 |
| famhs | 0 | 11 | 2987/25960612 | 5635031/5765681 | 226 | 6 | 10.328 |
| famhs | 1 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| famhs | 2 | 49 | 7435/115712051 | 22389628/22574200 | 8 | 0 | 36.316 |
| hsrand | 0 | 6 | 303/14170639 | 3203486/3228751 | 0 | 0 | 36.319 |
| hsrand | 1 | 0 | 0/0 | 0/0 | 0 | 0 | — |
| hsrand | 2 | 60 | 7454/141724778 | 26911473/27259539 | 0 | 0 | 36.316 |
| stress | 0 | 21 | 24939/49396786 | 9073845/10282058 | 3 | 3 | 28.334 |
| stress | 1 | 18 | 40206/42196883 | 6540485/7344465 | 9 | 9 | 32.678 |
| stress | 2 | 25 | 56854/58782098 | 9961245/10483334 | 21 | 11 | 35.492 |
| k30 | 0 | 71 | 1285/2736202 | 335119/345872 | 31 | 0 | 0.586 |
| k30 | 1 | 7 | 71/278664 | 30879/32574 | 0 | 0 | 0.586 |
| k30 | 2 | 2 | 20/79206 | 9875/10059 | 0 | 0 | 0.587 |

**Supplementary Table S5. Supported marginal-edge product against simulation truth.** Only named,
supported edges in the partial Tier-B table count. True parents outside the sequenced sample set
are excluded. This is the edge product used by phase and recombination, and can contain more edges
than the exact-configuration call product scored in main-text Table 3.

| Scenario | Supported edges | Correct supported edges | Extra supported edges | Missing supported edges |
| --- | --- | --- | --- | --- |
| std | 60 | 60 | 0 | 0 |
| seed | 60 | 60 | 0 | 0 |
| n80 | 120 | 120 | 0 | 0 |
| backcross | 0 | 0 | 0 | 120 |
| bc8 | 120 | 120 | 0 | 0 |
| longbc | 118 | 118 | 0 | 2 |
| f3 | 0 | 0 | 0 | 0 |
| longf3 | 0 | 0 | 0 | 0 |
| f3x8 | 0 | 0 | 0 | 0 |
| sibrich | 0 | 0 | 0 | 0 |
| tl | 80 | 80 | 0 | 0 |
| famhs | 98 | 98 | 0 | 0 |
| hsrand | 120 | 120 | 0 | 0 |
| stress | 68 | 68 | 0 | 0 |
| k30 | 11 | 11 | 0 | 29 |


## TroMau chromosome summaries

**Supplementary Table S6. TroMau QC and founder release.** SNP rows are those written by the
full-length export, including rows removed by site filtering; kept sites are the inference set.
Blocks are 200-site discovery blocks, including the final partial block. Sample QC flags affect
founder construction and do not remove those samples from painting or pedigree inference.
Release segments are counted from the export manifest; they need not equal the number of
source assembly components. Founder labels reset between segments.

| Chromosome | SNP rows | Kept sites | Blocks | Flagged samples | Release segments | Founder rows per segment |
|---|---:|---:|---:|---:|---:|---|
| chr1 | 233533 | 93460 | 468 | 1 | 1 | 8 |
| chr2 | 254453 | 89047 | 446 | 1 | 1 | 8 |
| chr3 | 749986 | 128257 | 642 | 1 | 1 | 8 |
| chr4 | 248773 | 81564 | 408 | 1 | 1 | 8 |
| chr5 | 232844 | 87364 | 437 | 1 | 1 | 9 |
| chr6 | 269001 | 97524 | 488 | 1 | 1 | 8 |
| chr7 | 422133 | 166616 | 834 | 1 | 1 | 8 |
| chr8 | 145568 | 58101 | 291 | 1 | 1 | 9 |
| chr9 | 286666 | 80058 | 401 | 1 | 1 | 8 |
| chr10 | 210484 | 83637 | 419 | 1 | 1 | 9 |
| chr11 | 233185 | 90287 | 452 | 1 | 1 | 7 |
| chr12 | 253473 | 86871 | 435 | 1 | 1 | 8 |
| chr13 | 195764 | 76025 | 381 | 1 | 1 | 9 |
| chr14 | 246481 | 92774 | 464 | 1 | 1 | 9 |
| chr15 | 229857 | 90025 | 451 | 1 | 1 | 7 |
| chr16 | 218895 | 78525 | 393 | 1 | 1 | 8 |
| chr17 | 236718 | 94486 | 473 | 1 | 1 | 8 |
| chr18 | 255678 | 83141 | 416 | 1 | 1 | 8 |
| chr19 | 210916 | 71342 | 357 | 1 | 1 | 8 |
| chr20 | 204509 | 74372 | 372 | 1 | 1 | 8 |
| chr22 | 261928 | 89376 | 447 | 1 | 1 | 8 |
| chr23 | 318039 | 102315 | 512 | 1 | 3 | 8,8,8 |


## The 320-sample depth series

The separate series uses `configs/simulation.toml` at release 4.4.0 (commit `867f163`), seed
400 and the empirical six-haplotype templates. All 320 animals in cohorts [20, 100, 200] are
observed; their template founders are unobserved. There are 22 contigs: chr1–chr20, chr22 and
chr23, in that order. Read depth is the only changed generating parameter (1, 1.5, 2 or 5).
Reads use the default independent-site Poisson model, error probability 0.02, heterozygous ALT
probability 0.5, no depth variation and no dropout. Generating and inference fallback rates are
both 5 cM/Mb, with no map files and the inference fallback marked unmeasured. Backcrossing is
off. Depths 1 and 1.5 additionally set the execution-only `memory_limit` to `250G`, which the
runtime interprets as 250 GiB. The runs use one seed and one replicate per depth; no cross-depth
hardware or lane-count matching was performed.

**Supplementary Table S7. The full depth series.** Panel A gives rates; panel B gives exact final
founder-refit and genotype counts; panel C gives phase and pedigree; panel D gives computation. Main-text
Table 6 reports the corresponding rates. All founder rows are matched, so using matched called
alleles (the manuscript definition) or all called alleles (the comparison script's definition)
gives the same one-to-one error rate here. Founder missing rates divide missing by called plus
missing alleles; the sum is 53,741,112 at every depth. Genotype rates divide errors by called
genotypes. Set-aligned site errors count minority-orientation sites within each released set
and divide by phased true heterozygotes; they are not switch events. Phased share divides those
heterozygotes by correctly called true heterozygotes. N50 is the global released-set span N50
from evaluation totals, recomputed from pooled spans by the evaluator, not averaged over
chromosome N50s.

::: {.depth-table}

**A. Rates and release.**

| Depth | Founder wrong / missing per $10^6$ | Genotype errors per $10^6$ | Set errors per $10^6$ | Phased share |
| --- | --- | --- | --- | --- |
| 1 | 5.55 / 19.05 | 115 | 68 | 0.9976 |
| 1.5 | 1.51 / 5.34 | 107 | 66 | 0.9976 |
| 2 | 0.54 / 2.20 | 98 | 50 | 0.9975 |
| 5 | 0.20 / 0.02 | 63 | 60 | 0.9913 |

:::

::: {.depth-table}

**B. Founder and genotype counts.**

| Depth | Called founder alleles | Wrong | Missing | Called genotypes | Genotype errors |
| --- | --- | --- | --- | --- | --- |
| 1 | 53,740,088 | 298 | 1024 | 2,864,927,399 | 330,862 |
| 1.5 | 53,740,825 | 81 | 287 | 2,864,738,842 | 305,481 |
| 2 | 53,740,994 | 29 | 118 | 2,864,497,726 | 280,489 |
| 5 | 53,741,111 | 11 | 1 | 2,864,012,489 | 179,900 |

:::

::: {.depth-table}

**C. Phase and pedigree counts.**

| Depth | Phased / correct true het. | Set site errors | Switches / long | Released sets | Released-set N50 (bp) |
| --- | --- | --- | --- | --- | --- |
| 1 | 487,466,845 / 488,652,509 | 32,979 | 31 / 13 | 7,101 | 39,779,055 |
| 1.5 | 487,541,151 / 488,701,752 | 32,028 | 27 / 7 | 7,118 | 39,779,037 |
| 2 | 487,471,374 / 488,707,602 | 24,468 | 20 / 6 | 7,114 | 39,779,055 |
| 5 | 484,309,497 / 488,570,031 | 29,210 | 19 / 7 | 7,143 | 39,759,093 |

:::

Every row has 600/600 true edges correct, no extra or missing edge, 320/320 states correct
and 320/320 exact configurations. The expected states are 20 zero-observed-parent and 300
two-observed-parent states; no sample has exactly one observed parent.

::: {.depth-table}

**D. Run-command computation.**

| Depth | Run (h:mm:ss) | Peak live (GiB) | Limit (GiB) | GNU max. RSS (GiB) | Processor / cores |
| --- | --- | --- | --- | --- | --- |
| 1 | 5:56:02 | 225.2 | 250.0 | 4.91 | Sapphire Rapids / 112 |
| 1.5 | 5:46:27 | 225.9 | 250.0 | 5.81 | Sapphire Rapids / 112 |
| 2 | 7:34:09 | 176.0 | 249.9 | 5.42 | Ice Lake / 76 |
| 5 | 4:44:29 | 254.8 | 500.4 | 5.34 | Sapphire Rapids / 112 |

:::

Peak live and limit columns convert `memory.peak_live_use_bytes` and `memory.limit_bytes` from
the completed `run` record using $2^{30}$ bytes/GiB. The peak is sampled job memory pressure,
including concurrent workers and possible node pressure, excluding reclaimable inactive file
cache. It is neither a sum of worker RSS nor a minimum required capacity. The limit is the
recorded governor bound, which need not equal a nominal node label. The two configured 250 GiB
limits were applied on larger nodes; depth 2 used the nominal 250 GB node. GNU time's maximum
process RSS converts KiB using $2^{20}$ KiB/GiB and covers `run` only. Its value is distinct
from the maximum over the separate init, run and evaluate commands in the campaign comparison
helper. Wall time is GNU time's elapsed `run` command, including command overhead, with init and
evaluation excluded; the runtime's narrower internal wall timer is not substituted. All four
runs completed in one attempt, with zero memory waits and zero forced grants. Run admissions
were 422,645, 402,551, 380,994 and 301,699 in increasing depth order. No timing or memory
uncertainty can be estimated from these single observations on mixed hardware.

## Generating-model details

**Templates (OU20, OU21).** Windows start every 50,000 bp from 0 to the contig length and cover $[\text{start},\text{start}+100{,}000)$. For rows $k$ and $k'$ of consecutive windows, the mismatch percentage $\Delta$ is 100 times their expected allele mismatches, $\sum_j P_k(0)P_{k'}(1)+P_k(1)P_{k'}(0)$, divided by the number of shared discovery sites; the similarity $\sigma=0$ if $\Delta>50$, otherwise $2(50-\Delta)$, and the junction score is $100(\sigma/100)^2$. Paths are found by reverse dynamic programming, the first candidate winning exact ties. After each path, every row's node score in a used window falls by 10 times its similarity to the used row, with similarity $1-\min(1,2\delta/n)$ for $\delta$ expected mismatches over the window's $n$ discovery sites, normalised by the square roots of the diagonal, and each internal used edge falls by 10. Overlaps are merged with signed information $I(p)=\mathrm{sign}(p-\tfrac{1}{2})\,(1-H_2(p))$, where $H_2$ is the binary entropy, combined as $(I_1+I_2)/(1+I_1I_2)$ and inverted by bisection. A contig with fewer than two informative windows gets no template. The linker is compiled without fast-math reassociation (ledger L14), which matters only when templates are rebuilt.

**Founders (OU23).** Each haplotype makes one vector draw for its exact ties, in site order, and only when ties exist.

**Pedigrees (OU24, OU25).** In the cohort design, the master stream's first draw seeds the structural generator. Each cohort's structure is built before its seed block is drawn. In a backcross-eligible cohort every child consumes one structural uniform draw: below `backcross_fraction`, the first parent is drawn uniformly from the previous cohort and the second uniformly from that parent's two parents; otherwise two distinct previous-cohort members form an ordered pair (first parent gives the maternal haplotype). In an explicit table, founders have depth 0 and others one more than their deeper parent; individuals are simulated by depth in table order, without a structural draw, and the founder count must equal $H/2$. Generated identifiers (`Founder_i`, `F<cohort>_<i>`) are truth labels only.

**Meiosis (OU26).** Without a map, gaps of $\lceil\mathrm{Exp}(\text{mean }1/\gamma)\rceil$ bp run from the first marker until past the last. With a map, the Poisson mean is the distance from first to last marker, including fallback-rate tails beyond the map's knots. A zero distance or rate consumes no crossover draws, and a marker exactly at a crossover takes the new homologue. Mutation gaps are $\lfloor\mathrm{Exp}(\text{mean }1/\mu)\rfloor$ bp; each event flips the first marker at or after it, and duplicate indices flip once. The child stream draws the maternal gamete, then the paternal.

**Reads (OU27, OU28).** The independent-site sampler draws, in order: depth factors (only if $v>0$), the full $N\times L$ Poisson array, dropout starts (only if $w>0$), then binomial ALT counts per dosage class 0, 1, 2, skipping empty classes. The fragment sampler draws factors once. Then, per sample, it draws a Poisson fragment count over the span from $z_{\max}$ bp before the first marker to the last marker, uniform starts, lengths and homologues, followed by per-read imbalance and error draws. Dropout starts are drawn after all samples. The start rate is $\lambda u_i/(1-w/L)/\bar c$, where $\bar c$ is the mean of $\min(z,2\ell)$ over insert lengths $z$ at the quadrature points. Both samplers consume whole-contig arrays, so splitting by sample would change the draws.

**Streams (OU30).**

| Stream | Construction | Draws |
|---|---|---|
| founders | PCG64, `SeedSequence([seed, 2000000])` child $c$ | Tie resolution on contig $c$ |
| master | PCG64, seed | Structure seed (cohort design), then per cohort or depth a contig-major $C\times n$ child-seed block |
| structure | Mersenne Twister seeded by `master.integers(0, 2^31)` | Parent pairs and backcross choices |
| child | PCG64, child seed | Maternal then paternal gamete |
| observation | PCG64, `SeedSequence([seed, 3000000])` | Sequenced subset |
| reads | PCG64, seed + 1,000,000 | One seed per contig, for every contig |
| contig reads | PCG64, contig read seed | Whole-contig reads |

Here $C$ is the number of contigs and $n$ the cohort size.

**Truth (OU29).** Per contig: positions; founder haplotypes int8[$H$, $L$] (consecutive row pairs form the diploid founders); sequenced samples' alleles int8[$N$, $L$, 2] (maternal, paternal); ancestry chunks with integer $[\text{start},\text{end})$ bounds and maternal and paternal founder haplotype ($-1$ for none); and one row per gamete of a sequenced child (child, parent, slot, cohort index, float crossover positions). The run manifest holds the state, requested and realised seeds, contigs, source identity, sequenced sample identifiers, and the sequenced and generated pedigree tables. `simulation/truth_pedigree.csv` summarises the sequenced samples' true parents.

## Metric definitions and edge cases

**Selection (OU31).** The default `available` mode scores the latest founder product of each contig plus any completed pedigree, family phase and recombination products, listing missing downstream products as skipped. `all` scores every founder product present; explicit stages fail if a product is missing. A complete simulation manifest is required. The pedigree comparison stays genome-wide when contigs are restricted.

**Founder panels (OU32, OU33).** A stage's components must not overlap and must lie on truth markers, and every truth marker counts. If $K_c>H$, $K_c-H$ rows stay unmatched and their calls are reported as unmatched called alleles; if $K_c<H$, each unmatched founder costs $L_c$ missing alleles. When several assignments are equally optimal, assignment-derived counts and strata can differ between them; a pure row permutation changes no count. Per-million rates are null with a zero denominator. In the segment path, an equal switch or stay cost keeps the current label, and the first terminal minimum wins. An infinite switch cost reproduces the nearest-founder count. Component-wise counts can fall when a component is cut, because labels re-match on each piece; the split-charged cost rises by at most $\kappa$ per row.

**Ancestry and strata (OU34).** Founder $f$ is represented at site $j$ if $c_{fj}\ge1$, ignoring unknown ancestry. Chunks are half-open, except that the chunk ending at the last marker includes it. Per stratum, missing alleles are truth alleles minus correct and wrong matched calls, so they include uncalled alleles, unmatched founders and uncovered markers.

**Identity continuity (OU35).** Windows are consecutive blocks of 100 truth markers, the last possibly partial, each centred on its middle marker. A row is labelled in a window when it has $n\ge16$ calls there, the best founder has at most $\max(1,\lfloor0.05n\rfloor)$ mismatches, and the second best has at least 2 more. An assertion between consecutive labelled windows $w_1<w_2$ is direct if $w_2=w_1+1$ and gap-spanning otherwise. A window's copy class is $\min(2,\lfloor\text{median }c_{fj}\rfloor)$. A correct assertion takes the poorer class of its two windows; a wrong one takes the departing founder's class in $w_2$. An assertion is flagged when a pipeline identity flag of its row lies in (centre of $w_1$, centre of $w_2$]; flagged assertions are counted separately, never as errors avoided. A founder's eligible windows have median copy count $\ge1$ and Hamming distance $\ge3$ to every other founder. Consecutive eligible windows at most 3 apart are covered when one row labels the founder in both and labels no other founder in between. Coverage is also reported without flagged rows. Splitting rows or dropping assertions can raise the correct share without raising coverage.

**Phase (OU36; definition `released_phase_v3`).** A new set starts at each component change and after each released break (boundary $j$ lies between sites $j$ and $j+1$). Sites outside components belong to no set. A scored site has both alleles called, the true allele sum, a true heterozygote, set membership, a called heterozygote, and a displayed phase that is not undetermined. Released sets need only a called heterozygote released phased: a call 0|1 against truth 0/0 gives one genotype error, one released set and no scored set. N50 is 0 without a positive span; the phased share is null with a zero denominator.

| Orientations within one set | Scored sites | Site errors | Switch events | Long switches |
|---|---|---|---|---|
| `0 0 0 1 1 0 0 0 0 1` | 10 | 3 | 3 | 0 |
| 30 zeros, 2 ones, 25 zeros, 21 ones | 78 | 23 | 3 | 1 |
| Alternating at every site, length $n$ | $n$ | $\lfloor n/2\rfloor$ | $n-1$ | 0 for runs below the long-run threshold |

Parent groups count released post-guard Tier-B edges into each sample, not true or cohort-derived parents.

**Pedigree (OU37).** The expected state is the number of a sample's true parents that were sequenced. An unresolved call (state $-1$) is never state-correct. Extra edges are called minus correct; missing edges are true minus correct.

**Map (OU38).** Edges are keyed by (parent, child) names over the gametes of sequenced children; an inferred edge that is not a true gamete is counted and skipped. A true edge without crossovers still contributes expected counts, exposure, calls and widths. Hit rates and median widths are null without calls. Called and reported intervals both count a hit when a true crossover lies in $(\text{left},\text{right}]$.

## Cost exemptions and exceptions at $K=30$

| Decision | Retained behaviour | Cost | Meaning at $K=30$ |
|---|---|---|---|
| 0 | Rare context-coded local-selection screening | $O(K^5)$ per round | $K^5=24{,}300{,}000$, an algebraic factor, not a measured count or time |
| Count refits | Every eligible deletion and duplicate seed is fitted | $O(NLK^3)$ per round under quadratic fixed-$K$ work; $O(NLK^4)$ worst-case deletion pass | Up to 30 eligible fits per round; at most one more deletion round than accepted deletions; shadow budgets and continuation histories are separate |
| 0c | Bounded discovery and local-selection searches | Discovery $O(NLK^4)$; selection backstop $O(NLK^3(K+L))$; context snaps $O((K+L)^2)$ passes | $K^4=810{,}000$, $K^3=27{,}000$; full cost depends on $N$, $L$ and the search path |
| 0d | Optional pairwise IBD | $O(N^2L)$ | Independent of $K$; 3,123,750 unordered pairs at $N=2{,}500$; off by default |
| 0e (AJ17) | Assembly joins' structural screening | Cubic analytic bound for the adaptive route; quartic direct-query trend on a finite grid | Separate from count-refit work; emissions ≈18.6 GB at $N=2{,}500$, $m=2{,}000$ |
| 0e (RS05, RS07) | Refinement window fixing | Bounded only by the work account | Analytic work-account limit; polish ≈3.7 GB at $N=2{,}500$ |
| 0e (PR17) | Painting label-continuity checks | $O(K^4)$ checks | ≈$10^{11}$ gain operations at $N=2{,}500$ with 2,000 links |
| 0e (FR04) | Founder-refit ties | $O(K^4)$ flips and $O(K^5)$ swaps per phase; $O(K^4)$ over $K$ rounds via the direct two-row gain (L10) | $K^4=810{,}000$; $K^5=24{,}300{,}000$ per site and phase; these are analytic work factors, not measured iteration counts |

The kernel ladders do not certify these whole-search bounds. Painting continuity and founder-refit tie decisions do act on validation inputs; their observed small classes do not exercise their analytic worst cases. Refinement's work account bounds canonical window fixing. The join cost register distinguishes its cubic adaptive route from the direct-route finite-grid trend. Here $m$ is the join emission-axis size used by the cost model.

## Exact evidence storage and execution

For one sample and contig with $L$ sites and $D$ distinct integer REF/ALT pairs, use the smallest
unsigned code width $w$ that represents $D-1$: 1 byte through 256 pairs, 2 through 65,536, and
4 through 4,294,967,296. Each dictionary entry stores the two int32 counts and three float64
likelihoods (32 bytes); a dense cell uses 33 bytes, including its observation flag. Coding
requires $32D+wL+17<33L$ for that row and a strictly smaller aggregate container after all
fallback rows, 17 bytes of metadata per sample and the final eight-byte contig offset are
included. A repeated pair with different likelihood bits forces dense storage. The observation
mask follows exactly from the pair; founder exclusions are additional row flags. The original
dense kernels provide the triples and every proposed codebook is checked against all source
bits in bounded windows. This introduces no likelihood approximation, depth cap or new setting.
Encoding costs $O(NL\log L+\sum_i L\log D_i)$, with one sample's $O(L+D_i)$ planning scratch
at a time, independently of $K$. Consumer-local dense workspaces still contribute to peak memory.

Checkpoint windows decompress each required whole chunk into a bounded decoded cache before
NumPy slices it. The cache normally holds at most 1 GiB, with one oversized chunk allowed alone;
a miss additionally holds its transient compressed input. Requests that exceed the cache can
repeat whole-chunk decoding. CPU claims refresh at kernel and independent-section boundaries,
within the task ceiling, rather than inside a running native kernel. The memory request tables
grow under the interprocess lock, preserve slot indices, and do not impose a fixed request cap.
Starts are paced only above 90% measured use; idle reloadable state can be released under
pressure. Resource decisions do not select scientific candidates or alter update order.

## Tie quanta and execution units

For the shared decision kinds, $q_0=2^{\lceil\log_2(16\nu)\rceil}$ with noise motions $\nu$ of $5.5\times10^{-5}$ (NLL, improvement), $10^{-5}$ (allele step), $5\times10^{-6}$ (sample), $2\times10^{-6}$ (cell) and $10^{-6}$ (probability). The relative factor $\alpha$ is $2^{-30}\cdot2^{10}$ for NLL, $2^{-36}\cdot2^{10}$ for sample, and 0 otherwise. Assembly decisions use their own floors and factors, and fixed policies use a declared width (EC30–EC32). The main-text execution mechanisms correspond to the dependency graph and barriers (ENG-01, ENG-03), run modes (ENG-04), identities and resume (ENG-06–ENG-08), the append-only container (ENG-14), the CPU budget and pools (ENG-20, ENG-21), contig lanes (ENG-23), the memory governor (ENG-25) and the kernel cache (ENG-27).
