# Results

## Validation design and founder reconstruction

The 4.2 release matrix supplies the validation evidence for the reconstruction and release rules
described here; 4.3.0 retains that science and presents the tables in the output layout described
in Methods. The 4.2 release matrix contained 39 completed runs: 15 simulated datasets with
recorded truth,
two real-data inputs (TroMau and its three-chromosome subset), and 22 perturbations of those
inputs. Thirteen simulations were design cases and two were references for perturbation tests;
the perturbations are repeated analyses of the same data, not biological replicates. Each design
used one seed, with a second seed for the 40-sample cohort design. The matrix is a collection of
contrasting designs rather than a balanced factorial experiment or a depth-response study.

The 39-case 4.3.0 release matrix, comparing every output cell with its mapped 4.2 release matrix
value, is pending [TO FILL: outcome of that comparison]{.todo}.

Except for the $H=30$ stress test, simulations used six empirical founder haplotypes per
chromosome. Eight-chromosome and three-chromosome designs varied cohort structure, backcrossing,
which cohorts were sequenced and whether the pedigree was explicit. The two explicit pedigrees
contained 115 and 60 observed samples. Other simulations contained 40–112 sequenced samples;
the read-stress reference retained 64 of 80 generated individuals. Mean depth was 5 or 10.5,
with independent-site reads or 125 bp paired-end reads. The stress reference combined depth
variation, allelic imbalance, dropout, increased read error and partial sequencing, so its effects
cannot be assigned to any one perturbation. Generating and inference maps were configured
independently, including map-length experiments. Supplementary Table S3 gives every design,
seed and read setting.

Only the final founder-refit product was scored in these saved evaluations. We therefore report
endpoint accuracy (Figures 3 and 4; Table 1); the outputs do not establish a trajectory of accuracy
through discovery, selection and assembly. Founder errors were pooled as counts over chromosomes
within each run. One-to-one errors use a single assignment of reconstructed rows to truth within
each component. Segment errors permit a row to change its true-founder label along the component,
with an explicit switch penalty. Their difference measures a limitation that a single allele-error
rate would hide: locally accurate sequence can switch biological founder identity over long spans.

| Scenario | One-to-one errors / $10^6$ | Segment errors / $10^6$ | Truth-to-panel errors + missing | Label switches | Wrong continuity assertions |
| --- | --- | --- | --- | --- | --- |
| std | 30,360.1 | 851.8 | 460,128 | 654 | 1305 |
| seed | 26,887.9 | 252.2 | 413,364 | 202 | 361 |
| n80 | 453.1 | 36.4 | 12,752 | 18 | 25 |
| backcross | 53.5 | 31.5 | 897 | 5 | 9 |
| bc8 | 57.0 | 14.8 | 2,592 | 4 | 9 |
| longbc | 1,601.4 | 33.7 | 10,592 | 8 | 17 |
| f3 | 38,247.4 | 10.6 | 223,991 | 17 | 19 |
| longf3 | 64,503.9 | 132.7 | 378,175 | 112 | 169 |
| f3x8 | 25,241.8 | 126.2 | 368,319 | 216 | 347 |
| sibrich | 25,127.4 | 70.6 | 365,383 | 110 | 168 |
| tl | 6.0 | 4.3 | 227 | 1 | 1 |
| famhs | 560.4 | 75.9 | 18,133 | 21 | 53 |
| hsrand | 26,284.6 | 17.1 | 373,251 | 64 | 72 |
| stress | 273.3 | 27.2 | 7,114 | 7 | 17 |
| k30 | 3,348.6 | 1,975.9 | 55,020 | 23 | 106 |

Table: Endpoint founder accuracy for the 15 simulated datasets. One-to-one errors are per million
matched called founder alleles; segment errors are per million called founder alleles. The
truth-to-panel count includes missing alleles, unmatched truth founders and uncovered sites. Label
switches are changes in the segment-wise truth label, not inferred meiotic crossovers. Wrong
continuity assertions include adjacent and gap-spanning assertions, whether flagged or unflagged.
Scenario names are defined in Supplementary Table S3.

The explicit `tl` pedigree had 6.0 one-to-one errors per million matched called founder alleles,
whereas `std` had 30,360.1 and `longf3` had 64,503.9. Segment errors in those same runs were 4.3,
851.8 and 132.7 per million called alleles, respectively. The `std` panel contained 654 truth-label
switches and 1,305 wrong continuity assertions. In that run, 450,328 truth alleles were absent
from the sequenced descendants and 1,045,902 occurred in only one sequenced homologue. These
results show why a released founder row should be read together with identity flags and segment
boundaries. Recovering its local alleles does not establish a persistent chromosome-wide identity.

![Accuracy on all scored simulation runs. Founder and genotype errors, phase switches, phased
share, allele callability and exact-configuration pedigree edges are shown against recorded
truth. Perturbations of a dataset are grouped together and are not independent simulations.
K = 30 and the combined read-stress design are marked separately.](figures/fig3_simulation_accuracy.svg){#fig:simulation_accuracy width=100%}

![Final founder recovery by component. Each point describes the saved founder-refit endpoint;
no intermediate-stage trajectory is implied. One-to-one allele errors and founder counts expose
heterogeneity among chromosomes and designs.](figures/fig4_founder_endpoint.svg){#fig:founder_endpoint width=100%}

## Genotypes, phase and pedigree against truth

Genotype errors ranged from 7.6 to 811.3 per million called genotypes, and the phased share of
correctly called true heterozygotes ranged from 90.98% to 99.23% (Table 2). The combined stress
reference had the lowest phased share and 23 long switches. Release can preserve a high phased
share while breaking phase sets more often: the three-chromosome `backcross` run had a released
phase-set N50 of 4.716 Mb, compared with 36.165 Mb for `bc8`. These are within-run measures;
N50 values were not averaged across designs. Supplementary Table S4 stratifies phase by the
number of supported parent edges into each sample.

| Scenario | Genotype errors / $10^6$ calls | Phased share (%) | Switch errors | Long switches | Released phase-set N50 (Mb) |
| --- | --- | --- | --- | --- | --- |
| std | 178.6 | 97.22 | 19 | 1 | 36.316 |
| seed | 112.6 | 97.97 | 22 | 0 | 35.621 |
| n80 | 83.5 | 98.60 | 14 | 2 | 36.316 |
| backcross | 74.3 | 98.96 | 1 | 1 | 4.716 |
| bc8 | 81.6 | 98.57 | 3 | 0 | 36.165 |
| longbc | 67.7 | 98.25 | 1 | 1 | 37.426 |
| f3 | 7.6 | 99.23 | 21 | 0 | 17.508 |
| longf3 | 16.3 | 99.07 | 8 | 2 | 6.103 |
| f3x8 | 11.4 | 99.20 | 127 | 2 | 11.523 |
| sibrich | 9.4 | 98.95 | 23 | 2 | 15.518 |
| tl | 72.5 | 98.39 | 5 | 1 | 17.638 |
| famhs | 73.6 | 98.89 | 234 | 6 | 35.621 |
| hsrand | 49.8 | 98.78 | 0 | 0 | 36.316 |
| stress | 811.3 | 90.98 | 33 | 23 | 32.399 |
| k30 | 444.7 | 96.75 | 31 | 0 | 0.586 |

Table: Genotype and phase accuracy. The phased share is scored phased true heterozygotes divided
by correctly called true heterozygotes. Switch errors are orientation changes between consecutive
scored sites within released phase sets; long switches require flanking orientation runs of at
least 20 scored sites. N50 uses released phase-set spans and is in megabases. Rates are computed
from pooled chromosome counts within each run.

The primary pedigree evaluator scores the exact-configuration Tier-B call product (Table 3).
The separate supported-edge product includes marginal edges from incompletely resolved
configurations and supplies phase and recombination; Supplementary Table S5 scores those edges
against the same truth. No exact-call extra edge was observed in these simulations, but recovery
was strongly design-dependent. With three chromosomes, `backcross` released no edges and all 80
parent states were unresolved: 0 of 120 observable edges were recovered. With eight chromosomes,
`bc8` recovered all 120 and all 80 configurations exactly. The `longbc` map-support experiment
recovered 116 exact-call edges on three chromosomes, and 118 when marginal edges were included.
It differs in seed and map support, so it is not an isolated chromosome-count comparison.

| Scenario | Exact configurations / $N$ | Correct states / $N$ | Correct exact-call edges / true edges | Extra exact-call edges | Missing exact-call edges |
| --- | --- | --- | --- | --- | --- |
| std | 40/40 | 40/40 | 60/60 | 0 | 0 |
| seed | 40/40 | 40/40 | 60/60 | 0 | 0 |
| n80 | 80/80 | 80/80 | 120/120 | 0 | 0 |
| backcross | 0/80 | 0/80 | 0/120 | 0 | 120 |
| bc8 | 80/80 | 80/80 | 120/120 | 0 | 0 |
| longbc | 77/80 | 79/80 | 116/120 | 0 | 4 |
| f3 | 5/112 | 5/112 | 0/0 | 0 | 0 |
| longf3 | 34/112 | 34/112 | 0/0 | 0 | 0 |
| f3x8 | 98/112 | 98/112 | 0/0 | 0 | 0 |
| sibrich | 96/112 | 96/112 | 0/0 | 0 | 0 |
| tl | 115/115 | 115/115 | 80/80 | 0 | 0 |
| famhs | 60/60 | 60/60 | 98/98 | 0 | 0 |
| hsrand | 66/66 | 66/66 | 120/120 | 0 | 0 |
| stress | 64/64 | 64/64 | 68/68 | 0 | 0 |
| k30 | 8/80 | 24/80 | 4/40 | 0 | 36 |

Table: Accuracy of exact-configuration Tier-B calls. True parents are restricted to sequenced
samples, and a configuration must have both the correct observed-parent state and the correct
parent set to count as exact. Unresolved states are never exact. A stable state can lack a
resolved identity; consequently correct-state counts can exceed exact-configuration counts.
These edges exclude the additional marginal edges tabulated in Supplementary Table S5.

In the omitted-parent cohort designs (`f3`, `longf3`, `f3x8` and `sibrich`), every observed sample's
parents were unobserved. Correct zero-observed-parent configurations ranged from 5 of 112 to 98
of 112, with many unresolved states. In the short $H=30$ design, only 24 of 80 states and 8
configurations were correct; exact calls recovered 4 of 40 observable edges. Including marginal
edges recovered 11, leaving 29 missing. Good genotype calls in this design therefore do not
establish a well-resolved pedigree.

Recombination evaluation is conditional on inferred edges that correspond to true gametes.
For `std`, 737 of 740 called intervals contained a true crossover (99.59%); the reported
intervals also contained 737 hits. There were 790 true crossovers within inferred informative
exposure, against a model expectation of 757.56. On chr1 of that run, the median called interval
was 75.15 kb wide and the reported interval 89.29 kb, with 106 hits among 107 calls. The combined
stress reference had 874 hits among 912 called intervals (95.83%), and 876 among its reported
intervals. Width and informative exposure qualify these hit rates: they do not measure recovery
of every crossover in the full pedigree or unobserved genome. Per-chromosome counts and widths
are retained in the evaluation tables.

Per-sample local-ancestry paintings and relatedness families have no direct truth score in the
saved evaluator. Painting is assessed through the relabelled founder panel, and relatedness
families through downstream pedigree output; neither supports a claim of direct ancestry or
sibship accuracy.

## Stability under perturbation

The robustness analyses permuted sample order, perturbed observed genotype likelihoods by
relative factors at scales $10^{-12}$ and $10^{-9}$, applied value-consistent $10^{-12}$ noise,
and reversed arbitrary tie conventions (Figure 5). Real-input tests also permuted the VCF
sample axis. Likelihood triples were renormalised, read counts were unchanged and zero-depth
cells remained exactly uniform. Each stochastic perturbation used one fixed seed. These are
numerical and ordering tests, not independent sequencing or biological replicates.

All available numerical-noise and sample-order variants agreed with their references in the
compared product families after aligning sample identifiers and canonicalising display conventions.
The numeric comparison used relative tolerance $10^{-9}$ and absolute tolerance $10^{-8}$;
phase and genotype calls in the aligned VCF records were compared exactly. This agreement
covers the tested perturbation scales and seeds, not arbitrary perturbations.

Tie reversal exposed stronger sensitivity. The three-chromosome actm3 subset agreed throughout,
but the n80 and stress designs differed in phase or recombination products, and TroMau differed
in founder, painting, phase and recombination products. On TroMau chr3, reversal retained three
release segments rather than the reference's single eight-founder segment. Its primary Tier-B
states and unordered parent sets were unchanged; parent-slot swaps carry no biological meaning.
Figure 5's broader pedigree-family comparison includes other rendered tables. A convention-sensitive
founder structure is not a uniquely determined biological result.

![Agreement of perturbations with their own dataset's reference run, by output family. Samples,
relatedness-family identifiers and unordered parent slots are aligned; painting track order is
canonicalised within sample and component. Numeric cells use relative tolerance $10^{-9}$ and
absolute tolerance $10^{-8}$. VCF comparisons align sample columns and retain founder labels,
calls, missingness, phase and PS, excluding metadata headers. Founder identities and phase
orientations are not rematched under reversal. Grey marks indicate absent variants. No real-data
output is compared with biological truth.](figures/fig5_perturbation_agreement.svg){#fig:perturbation_agreement width=100%}

## Inference on the TroMau cross

TroMau contains 116 sequenced samples on 22 chromosomes. The configured input is a multi-sample
BCF with allele depths; 5,918,884 SNP rows reached the site filter, of which 1,995,167 were kept
(58,101–166,616 per chromosome). Full-length exports retain removed rows with filter annotations. Supplementary Table S6 gives
the per-chromosome site, block, QC and founder-release counts.
The cross involves *Astatotilapia calliptera* and a *Tropheops* species
[TO FILL: precise species identification and experimental sampling description]{.todo}.
[TO FILL: sequencing platform, library preparation and variant-calling protocol beyond the recorded BCF fields]{.todo}.
The four samples described as founding-generation animals comprise three *A. calliptera* females
and one *Tropheops* male. Generation and sex describe the dataset; no sequenced individual is a
known parent, and neither these labels nor species assignments entered inference.

The configuration supplied no map file and used a uniform 5 cM/Mb fallback, explicitly marked
unmeasured. It therefore supplies model distances but no measured genetic length. Each chromosome
contributes one map-support unit to relatedness and pedigree release. Recombination estimates
below remain conditional on this map, the inferred pedigree and released phase.

The final panel had one component on 21 chromosomes, with chr23 split into three. Chromosome-level founder counts were eight on 15 chromosomes, nine on five and seven on two;
each of chr23's three components had eight rows. The founder mosaics and chromosome summaries show both the shared templates and boundaries that
qualify their identity (Figures 6–8). The relatedness fit assigned 72 families: 48 singletons,
15 pairs, six triplets and one family each of sizes four, seven and nine. These are inferred
families, not documented sibships.

Tier B contained 69 edges in exact configurations; including a marginal edge from its unresolved
sample gave 70 supported edges, used downstream. Twenty-two samples had two supported parents,
26 had one and 68 had none. The exact view resolved 68 samples as having zero observed parents,
25 as having one and 22 as having two; one remained unresolved (Table 4; Figure 9). A
zero-observed-parent state does not identify a founding-generation animal.

| View | Edges | Samples with two / one / no named parents | Zero / one / two observed-parent states | Unresolved states |
|---|---|---|---|---|
| Tier A exact configurations | 66 | 21 / 24 / 71 | 68 / 24 / 21 | 3 |
| Tier B exact configurations | 69 | 22 / 25 / 69 | 68 / 25 / 22 | 1 |
| Complete acyclic best guess | 70 | 22 / 26 / 68 | 68 / 26 / 22 | 0 |

Table: TroMau pedigree inference. The complete acyclic view is diagnostic and is not an additional
release tier. Tier-B supported marginal edges bring its total to 70 even though one sample's
parent count remains unresolved. Parent slots are unordered and do not encode sex. No entry is
an accuracy estimate or a comparison with cohort-derived parentage.

All 115 samples with resolved Tier-B states had informative evidence on all 22 chromosomes.
Their state-winner margins ranged from 3.26 to 43.96 nats; finite conditional identity margins
ranged from 13.67 to 51.94 nats. Across the 70 supported edges, chromosome-bootstrap support
ranged from 0.770 to 1.000 and LOCO support was 1.000 throughout. These are stability measures
under the composite inference model. They do not calibrate the probability that a particular
parent assignment is biologically correct.

Family phase released 96,715,320 phased and 5,066,079 unphased heterozygous calls at kept sites:
95.02% of called heterozygotes were phased. The chromosome-level share ranged from 86.42% to
97.98%, and 26,869 phase sets were reported. These summaries describe release, not accuracy.
The 70 supported meioses on each chromosome yielded 874 called crossovers over all chromosomes,
with 23–94 per chromosome. The exposure-normalised cumulative map lengths ranged from 38.87 to
159.29 cM, summing to 1,476.75 cM over covered sequence. The report-only detector found 40
junction loci, five flagged; these flags annotate interpretation and did not mask or refit maps.

![TroMau chromosome 1 founder paintings for all 116 samples, with two tracks per sample.
Track order is an arbitrary painting frame and does not assign maternal or paternal origin.
Colours identify component-local rows H0–H7; hatched grey pools local equivalence, posterior
abstention, switch intervals and background. Samples are ordered by inferred relatedness families.
These are painting-stage intervals, distinct from the final family-phase VCF.](figures/fig6_tromau_mosaics.svg){#fig:tromau_mosaics width=100%}

![TroMau recombination along all 22 chromosomes, called crossovers summed per supported
meiosis, and chromosome-average calls per meiosis. Rates are neither smoothed nor rebinned;
hatched bins have no rate estimate. The 70 supported chains include one marginal edge beyond
the exact-configuration pedigree in Figure 9. Estimates condition on inferred pedigree, phase,
informative exposure and the fallback map; they do not count every biological crossover.](figures/fig7_tromau_recombination.svg){#fig:tromau_recombination width=100%}

![TroMau founder counts and component structure by chromosome, including the convention-sensitive
chr3 comparison. Founder-row counts describe released components and need not equal a known
number of breeding founders.](figures/fig8_tromau_founder_counts.svg){#fig:tromau_founder_counts width=100%}

![TroMau inferred family sizes, Tier-B parent states and exact-configuration parent–child edges,
with selected graph-configuration bootstrap and LOCO support. The edge display contains 69 edges
for 47 children and 14 inferred observed parents. It excludes the additional marginal edge used
downstream. Configuration support is distinct from edge support; neither is calibrated accuracy.
Cohort labels and sample names supply no parentage truth.](figures/fig9_tromau_pedigree.svg){#fig:tromau_pedigree width=100%}

## Computation and the cost of complete count-candidate refits

Kernel ladders (Figure 2; Supplementary Table S1) measured direct synthetic kernels, not complete
adaptive searches. Every kernel gated as at most cubic had a fitted $K$ slope below 3.3. The
assembly site posterior had slope $3.06\pm0.11$, family-phase posterior $2.81\pm0.02$, and
context matching in local selection $4.01\pm0.05$; the latter belongs to the documented
$O(K^5)$ context-screening exemption. Relatedness pair screening and pedigree pair and family
kernels scale approximately quadratically in sample count. These observations support the
kernel models without certifying whole-stage cubic cost.

The 4.2 release matrix measurements include compilation, inference, reports and export, with
initialisation
reported separately (Table 5; Figures 10 and 11). Node types differ across the matrix, so cross-case and
cross-release wall-time ratios are not controlled speed comparisons. The representative TroMau
run used a full 76-core Intel Ice Lake node; `std` used a full 112-core Intel Sapphire Rapids
node. Memory columns deliberately distinguish GNU time's maximum process RSS from the runtime's
sampled job memory-pressure peak. The latter includes the job cgroup and concurrent workers,
excludes reclaimable inactive file cache, and can also reflect pressure on available node memory.
It is not a sum of per-task peak RSS values.

On a full 112-core Sapphire Rapids node, the same TroMau analysis from the 4.2 release matrix
took 68.57 min for `run`
and 96.36 s for `init`, with maximum process RSS of 15.45 GB; its released outputs equal those
of the Ice Lake matrix reference, with elapsed-time fields excluded from the comparison.

| Input | Processor / cores | Run (min) | Init (s) | Maximum process RSS (GB) | Sampled pressure peak (GB) |
|---|---|---|---|---|---|
| TroMau, 116 samples, 22 chromosomes | Ice Lake / 76 | 102.43 | 3.52 | 15.39 | 121.86 |
| TroMau, same input, timing probe | Sapphire Rapids / 112 | 68.57 | 96.36 | 15.45 | 170.54 |
| `std`, 40 samples, eight chromosomes | Sapphire Rapids / 112 | 15.92 | 21.13 | 13.33 | 132.38 |
| $H=30$, 80 samples, five short contigs | Ice Lake / 76 | 54.32 | 17.13 | 1.45 | 104.42 |

Table: Representative whole-run measurements from the 4.2 release matrix and the matching
TroMau timing probe; these are validation measurements, not new 4.3.0 benchmarks. Time is elapsed
wall time for `run`, with `init`
separate and truth evaluation excluded. RSS is GNU time's `Maximum resident set size`, converted
from KiB to decimal GB. The pressure peak is `memory.peak_live_use_bytes` in the completed run
record, also in decimal GB. Neither column is interchangeable with the other or a projected
minimum memory requirement.

The 4.2 release matrix comparison with its recorded baseline used identical 76-core Ice Lake nodes
and the same input,
configuration and seed for $K=30$ (Figure 11). Complete count-candidate refitting took 52.63 min versus
36.17 min for the baseline, a 1.46-fold wall-time cost. Maximum process RSS was 1.49 versus 1.45 GB.
On a saved $N=2,503$ pedigree decision input, using 16 cores on the same node, the decision took
395.16 versus 433.37 s, the decision plus reports 405.34 versus 475.58 s, and maximum RSS
9.97 versus 12.89 GB. Rendered tables agreed. The pedigree replay used an integrated development candidate and is not a new benchmark
of the final whole-genome release. This bounded
decision-step comparison does not
establish a whole-genome speedup at that sample count. The top-level GNU time user/system
fields do not account for all forkserver work, so they are not reported as total pipeline CPU time.

![Whole-run wall time from the 4.2 release matrix and its baseline, separated by processor
architecture.
Each point is an observed run; inputs and node types vary. Initialisation and evaluation are
separate from the plotted inference command.](figures/fig10_whole_run_cost.svg){#fig:whole_run_cost width=100%}

![Cost comparisons for the K = 30 simulation, the saved N = 2,503 pedigree decision step and
the TroMau processor timing probe. The K = 30 runs use identical node types; the pedigree
comparison uses the same node and 16 cores. TroMau compares the 4.2 release matrix Ice Lake
reference (76 cores) with Sapphire Rapids (112 cores), with matching released outputs. Times
cover `run` only; memory is GNU time maximum process RSS.](figures/fig11_matched_cost.svg){#fig:matched_cost width=100%}

The $K=30$ endpoint comparison between the 4.2 release matrix and its baseline also illustrates
the scientific effect of fitting all eligible
count candidates (Figure 12). On chromosome m2, the released panel had 30 rows rather than 32,
and called-allele errors fell from 1,431 to 603. Genotype errors over the design fell from 1,387
to 1,376 and the phased share rose from 96.57% to 96.75%, while switch errors changed from 30
to 31. Pedigree recovery remained weak. Across the release comparison, 32 of 39 runs retained
the same released products; the scientific changes were confined to this simulation and six
TroMau analyses, including its reference. TroMau's Tier-B pedigree was unchanged. These
comparisons describe the 4.2 release matrix evidence used here; they are not a new 4.3.0
validation or a competing-method accuracy benchmark.

![K = 30 founder recovery and secondary accuracy measures from the 4.2 release matrix and
its recorded baseline. Chromosome m2 releases
the true number of founder rows with fewer called-allele errors; genotype, phase and pedigree
panels show the accompanying trade-offs.](figures/fig12_k30_recovery.svg){#fig:k30_recovery width=100%}
