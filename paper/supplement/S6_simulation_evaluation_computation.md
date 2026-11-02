# S6. Simulation, evaluation and computation

## Simulation settings and constants

Table S1 lists every simulation setting with its default. Settings in `[simulation]` and `[simulation.reads]` describe the generating model only; inference keeps its own read model and map. The order of `[input] contigs` is part of the random streams, and every contig's template must hold the same even number $H$ of haplotypes.

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

Table S2 lists the internal constants, which are not user settings.

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

The `haplotypes example` generator (OU22) writes three contigs of 1,200 evenly spaced sites over 12,000,000 bp with six independent Bernoulli haplotypes, cohorts (20, 30, 30) and depth 5.0; it demonstrates software wiring only. Two simulation configurations are shipped: `configs/simulation.toml` (22 contigs, chr1–chr20, chr22 and chr23; seed 400; cohorts [20, 100, 200]; depth 5.0; generating and inference fallback rates 5.0 cM/Mb) and `configs/simulation_backcross.toml` (the same contigs; seed 6202; cohorts [20, 30, 30]; `backcross_fraction` 0.5 from cohort 3). [TO FILL: list the configurations, seeds and read-model settings of the simulations reported in the Results.]{.todo}

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
| 0b | Ranked assembly and refinement seed refits (`ranked_refits` cap) | Fixed admission cap | The cap does not grow with $K$, so candidates can be omitted at larger $K$; costed as 9 ranked refits and at most 2,048 shadow scores per member |
| 0c | Bounded discovery and local-selection searches | Discovery $O(NLK^4)$; selection backstop $O(NLK^3(K+L))$; context snaps $O((K+L)^2)$ passes | $K^4=810{,}000$, $K^3=27{,}000$; full cost depends on $N$, $L$ and the search path |
| 0d | Optional pairwise IBD | $O(N^2L)$ | Independent of $K$; 3,123,750 unordered pairs at $N=2{,}500$; off by default |
| 0e (AJ17) | Assembly joins' structural screening | Quartic trend in $K$ of the direct query count on a finite grid | Analytic worst case; emissions ≈18.6 GB at $N=2{,}500$, $m=2{,}000$ |
| 0e (RS05, RS07) | Refinement window fixing | Bounded only by the work account | Analytic worst case; polish ≈3.7 GB at $N=2{,}500$; provisional |
| 0e (PR17) | Painting label-continuity checks | $O(K^4)$ checks | ≈$10^{11}$ gain operations at $N=2{,}500$ with 2,000 links |
| 0e (FR04) | Founder-refit ties | $O(K^4)$ flips and $O(K^5)$ swaps per phase; $O(K^4)$ over $K$ rounds via the direct two-row gain (L10) | $K^4=810{,}000$; $K^5=24{,}300{,}000$ per site and phase; observed at most 3 rounds and 2 tie members |

None of the 0e worst cases binds on the validation inputs; all four are scheduled for review. The join size $m$ is as written in the cost table, whose symbol list does not define it.

## Tie quanta and execution units

For the shared decision kinds, $q_0=2^{\lceil\log_2(16\nu)\rceil}$ with noise motions $\nu$ of $5.5\times10^{-5}$ (NLL, improvement), $10^{-5}$ (allele step), $5\times10^{-6}$ (sample), $2\times10^{-6}$ (cell) and $10^{-6}$ (probability). The relative factor $\alpha$ is $2^{-30}\cdot2^{10}$ for NLL, $2^{-36}\cdot2^{10}$ for sample, and 0 otherwise. Assembly decisions use their own floors and factors, and fixed policies use a declared width (EC30–EC32). The main-text execution mechanisms correspond to the dependency graph and barriers (ENG-01, ENG-03), run modes (ENG-04), identities and resume (ENG-06–ENG-08), the append-only container (ENG-14), the CPU budget and pools (ENG-20, ENG-21), contig lanes (ENG-23), the memory governor (ENG-25) and the kernel cache (ENG-27).
