# Validation

What has been validated, how, and the known limitations. Simulations with a known pedigree give supervised
accuracy; real crosses have no individual-level parentage or founder truth, so they are judged by consistency,
stability and agreement with the previous release, never by accuracy against an assumed truth.

## Acceptance of release 2.0

Release 2.0 rebuilds release 1.0 (commit 92b3daf) with the same science. It is accepted when:

1. **Science.** On simulations, accuracy against the known truth is at least as good as release 1.0 on every
   evaluated metric, judged against two release-1.0 replicates (seeds 10000 and 10001); on the real TroMau cross,
   results agree with release 1.0 apart from documented fixes, and the TroMau benchmark scorecards are no worse. Rounding-level differences are expected; a difference in a discrete outcome (a founder count, a
   pedigree call, a phase switch) must be explained.
2. **Performance.** The whole-genome seed-10000 simulation on a dedicated 112-CPU Sapphire Rapids node is no slower
   and uses no more peak memory than release 1.0 (within run-to-run noise); the TroMau cross likewise.
3. **Determinism.** Results do not change with `--cores` or the number of contig lanes, or with memory pacing.
4. **Founder counts.** Up to K = 30 founders per block run without hidden caps, and no stage's cost grows
   exponentially in K (synthetic kernel benchmarks at K = 12, 20 and 30).

### Scenarios

| scenario | design | contigs |
| --- | --- | --- |
| standard N80 | simulation: cohorts 20, 30, 30 at 5x | 8 |
| stress N80 | simulation: cohorts 20, 30, 30; half of the third cohort backcrossed, 80% observed, depth CV 0.6, 10% dropout, allelic bias 0.6, error 0.04 | 8 |
| F3 families | simulation: cohorts 16, 4, 96 (cohorts 1 and 3 observed), 10.5x paired-end reads, 2 cM/Mb | 6 |
| real-like | simulation: cohorts 16, 96, 10.5x paired-end reads, 2 cM/Mb | 6 |
| TroMau, three contigs | real cross (chr8, chr13, chr19), no metadata | 3 |
| whole genome N320 | simulation: cohorts 20, 100, 200 at 5x, seeds 10000 and 10001 | 22 |
| backcross N80 | simulation: cohorts 20, 30, 30, half of the third cohort backcrossed | 22 |
| TroMau | real cross, `tromau` preset | 22 |

Every simulation has six truth founders per block; scaling to larger founder counts is checked by the K-ladder
simulations. The AulStu cross is not used for validation runs. Release 1.0 and release 2.0 are compared in the
same software environment.

### Results

Release 2.0 is accepted: on all nine scenarios every difference from release 1.0 is explained by one of the
documented fixes ([changelog](../CHANGELOG.md#scientific-fixes)) or by rounding and exact ties. The final release
tree reproduces the accepted runs record for record (standard N80 and the whole TroMau genome), apart from the
export files it adds.

**Simulations.** Pedigree calls against the truth are identical on all seven simulations (exact configurations,
correct, extra and missing edges). Genotype and phase errors change by less than 5% of the difference between the two
release-1.0 seeds (3,182 genotype errors and 50 phase switch errors):

| scenario | exact configurations | correct / extra / missing edges | genotype errors, 1.0 -> 2.0 | phase switch errors, 1.0 -> 2.0 |
| --- | --- | --- | --- | --- |
| whole genome, seed 10000 | 320 | 600 / 0 / 0 | 481,081 -> 481,039 | 599 -> 598 |
| whole genome, seed 10001 | 320 | 600 / 0 / 0 | 477,899 -> 477,956 | 649 -> 651 |
| backcross N80 | 80 | 120 / 0 / 0 | 102,322 -> 102,326 | 151 -> 151 |
| standard N80 | 80 | 120 / 0 / 0 | 36,161 -> 36,161 | 39 -> 39 |
| stress N80 | 64 | 68 / 0 / 0 | 117,523 -> 117,524 | 93 -> 93 |
| F3 families | 54 | 0 / 4 / 0 | 11,727 -> 11,727 | 159 -> 159 |
| real-like | 111 | 190 / 0 / 2 | 12,252 -> 12,252 | 8 -> 8 |

Every changed metric sits on a contig where a fix acts: the read model's selection (most changes; it selects another
model on 34 of 94 simulated contigs), the Haldane painting map (stress N80 chr2: 100 genotypes released differently,
no genotype or phase change) or the family-phase polish's rigid first bin (backcross chr8: one sample's frame over
about 6 Mb where the data cannot tell which parent passed which haplotype, an exact tie broken the other way;
genotypes identical). Discovery itself is unchanged: every differing discovery block (909 traced over the nine
scenarios; for TroMau, its 8 contigs with an unchanged read model) is reproduced exactly by release 1.0's own
discovery run on release 2.0's evidence, and founder panels differ only where the evidence or discovery differs or
through the assembly and refit fixes.

**TroMau (whole genome, 116 samples).** Founder counts are identical on all 22 contigs (179 founder rows), and on the
three-contig set. Tier B releases 93 calls instead of 96 (34 parent edges instead of 39): the three calls release
2.0 leaves unreleased are near the release threshold (chromosome-bootstrap state fractions 0.51-0.62 against 0.70) and are exactly
the calls release 1.0 itself leaves unresolved without the homozygote-overdispersion read model, so they follow from
the read-model decision; no parent or edge appears that release 1.0 did not call. Against the TroMau benchmark
pedigree, calls agreeing exactly go from 93 to 90 and parents contradicting it stay at 3. The benchmark scorecards
stay within release 1.0's rounding floor (matched concordance 0.9582 -> 0.9571, Hungarian matching 0.8810 -> 0.8831,
best-row coverage 0.9226 -> 0.9236, changes 1,854 -> 1,795). Export cells differ where the evidence (14 of 22 contigs
select another read model), the founder refit's swap rule or the pedigree edges changed.

**Performance.** Whole genomes on dedicated 112-CPU Sapphire Rapids nodes with cold kernel caches
([performance](performance.md#reference-timings)):

| run | release 1.0 wall | release 2.0 wall | ratio | largest process, 1.0 -> 2.0 | node peak, 1.0 -> 2.0 |
| --- | --- | --- | --- | --- | --- |
| simulation seed 10000 | 2:08:46 | 2:03:41 | 0.961 | 27.4 -> 20.0 GiB | 302 -> 259 GiB |
| simulation seed 10001 | 2:13:49 | 2:09:56 | 0.971 | 28.5 -> 19.9 GiB | 313 -> 258 GiB |
| backcross N80 | 40:46 | 38:53 | 0.954 | 8.9 -> 5.9 GiB | 170 -> 136 GiB |
| TroMau (both runs at the same time, on two nodes) | 41:00 | 41:10 | 1.004 | 6.4 -> 5.3 GiB | 114 -> 96 GiB |

**Determinism.** The standard N80 simulation run by the final release tree at 48 cores is identical record for record
to the accepted run, which ran at 56 cores, was interrupted and resumed at 112 cores; earlier checks gave identical
records at 20 and 36 cores (standard N80), 8 and 38 cores (TroMau, three contigs) and one and two discovery threads.
Both releases were compared in the same software environment: release 1.0 rerun in a new environment with identical
genotype likelihoods already changes discovery in a few blocks (2-44 per bounded scenario, 364 of 9,979 TroMau
blocks) and one near-threshold TroMau Tier-B call, so references from another environment are not like for like.

Founder counts: on K-ladder simulations with K = 12, 20 and 30 distinct founders per block, every stage ran to
recombination with no backstop hit and no frontier fallback, and at N = 320 discovery recovered 12.02, 20.04 and
30.01 rows per block; costs per stage are in [performance](performance.md#founder-counts). With weakly supported
founders (N = 80, about five carriers per founder per block at K = 30) discovery found 25.5 rows per block and the
final components carried 29-30 founders; there the data-dependent search work grows faster than K^3.

## How differences are judged

Before comparing the two releases end to end, release 1.0's own sensitivity to rounding was measured: release 1.0
reran with every observed genotype likelihood perturbed by a relative 1e-12 or 1e-9 (and, on the three TroMau
contigs, with a smooth shift of the read model), and every discrete outcome was compared stage by stage with the
unperturbed run. Unperturbed reruns reproduce release 1.0 exactly, so every difference is caused by the
perturbation. On the standard N80 simulation such perturbations change a handful of the 11,822 discovery blocks and
at most a few founder cells after assembly, and no pedigree call, phase call, crossover call or headline metric. On
the real TroMau contigs, where many choices are near ties, they reach every stage: up to a few percent of
discovery blocks, most assembled components, and the founder count of one final component, while Tier-B pedigree
calls and crossover calls never changed. Differences between the releases within this floor count as rounding;
a zero-floor outcome that changes, or a count well above the floor, needs a root cause traced stage by stage
(typically by running release 1.0's downstream stage on release 2.0's upstream products).

## Stage-by-stage port checks

Each stage was first validated in isolation: release 1.0's inputs of the stage, converted to the new types, run
through the release-2.0 stage and compared with release 1.0's products on the bounded scenarios (and single contigs
of the whole-genome runs). The criterion is identical discrete outcomes (founder counts, row multisets, release
status, pedigree calls, genotype calls) except for demonstrated ties or rounding, and continuous values to
rounding where the algorithm is unchanged.

| stage | isolation check against release 1.0 | outcome |
| --- | --- | --- |
| evidence, simulation source | four simulated designs; the bounded sets and TroMau contigs | simulated reads and truth bit-identical; evidence identical apart from genotype-likelihood rounding of the single emission kernel (largest difference about 1e-10, inside release 1.0's own spread between optimiser starts) |
| discovery, local selection | TroMau chr8 (three-contig set), simulation seed 10000 chr19 (1,364 blocks), the standard N80 genome | discovery blocks and all three local-selection rounds identical (panels, frequencies, calls, objectives); genome-wide differences traced to the evidence rounding |
| assembly | standard N80 chr13, TroMau chr8 and chr13, simulation seed 10000 chr19, final and context assemblies | completion bit-identical; every hierarchy level identical (component boundaries, founder counts, row paths in release 1.0's order, plans) |
| founder refinement | 26 level runs: 678 components, 5,786 converged fits, 791 count decisions | 0 mismatches |
| painting, founder refit | all bounded sets and TroMau chr3, chr6 and chr8 (34 contigs, 7.87 million sample x bin cells) | release status, labels, classes, member sets, switch counts and refit cells identical; posteriors within 5.4e-15 |
| pedigree | the bounded sets; a decision replay of the whole TroMau genome from release 1.0's scores | calls identical; all 14 TroMau tables identical; one rounding-decided tie moved F3-family bootstrap diagnostics by 0.010 |
| family phase, recombination | the bounded sets | genotype calls, frames and recombination outputs identical |

The large-founder-count memory changes, the memory governor, the founder-refinement speed-ups and the clean-up after
the port were each checked for identical results on the bounded sets before they were adopted; the scientific fixes
were measured one by one against the unfixed tree (changelog).

## Known limitations

- Founder reconstruction can be ambiguous where ancestry is unsampled, where many descendants repeat one ancestral
  recombinant, or where two founders are identical over a region; the painting then pools rows it cannot separate
  ([painting](methods/painting.md)).
- Painting errors and structured missingness can affect pedigree direction; ambiguous, missing-parent and highly
  structured-missingness designs need validation by simulation before their calls are trusted.
- Tier-B pedigree calls need enough map length: five informative chromosomes shorter than 1 Morgan or without a
  genetic map (Tier A six), fewer long chromosomes on a supplied map; with less, no Tier-B parent is released, and
  family phase and recombination have no parent edges ([pedigree](methods/pedigree.md#map-length-requirement-release-32)).
- Support tiers, bootstrap fractions and posterior thresholds measure internal stability, not calibrated
  probabilities. The real crosses have no individual-level trio truth; a cohort label is not parentage.
- Family phase is conditional on the released Tier-B edges and the maps on the final phase and the fixed inferred
  pedigree; neither propagates pedigree uncertainty ([family phase](methods/family_phase.md),
  [recombination](methods/recombination.md)).
- The standard simulations have six founders per block; behaviour at larger founder counts rests on the synthetic
  benchmarks and the real crosses.

## Reproducing an evaluation

`haplotypes evaluate RUN --stages all` writes the simulation metrics of a run ([simulation.md](simulation.md#evaluation));
`haplotypes export` writes the founder and sample VCFs (with `--products tracks`, also the founder tracks) that the
cross-release comparisons use. Record the source commit, the run's `config.toml` and its run records with every
result ([simulation.md](simulation.md#recording-a-run)).
