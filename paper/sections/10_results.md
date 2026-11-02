# Results

## Accuracy on simulated crosses with recorded truth

We measured accuracy on simulated crosses generated from founder templates with recorded truth
(Methods), scoring each run with the evaluation described above. [TO FILL: the simulation matrix:
template source, contigs, founder haplotypes $H$, cohort sizes and designs (including backcrossing
and explicit pedigrees), fraction of individuals sequenced, depth, depth variation, allelic
imbalance, dropout, paired-end fragments, and seeds and replicates per cell.]{.todo}

Because every stage's founder product is scored on its own components, the accuracy of the
founder haplotypes can be followed through the pipeline, from the local panels of discovery to the
refitted chromosome-scale founders (Table 1). One-to-one errors, rather than nearest-founder
errors, are the primary measure, because duplicated rows can match the truth closely without
recovering every founder; the truth-to-panel count additionally charges founders that no row
recovers.

| Stage | One-to-one errors per $10^6$ called alleles | Truth-to-panel errors and missing calls | Segment label switches | Wrong identity-continuity assertions |
|---|---|---|---|---|
| Discovery | · | · | · | · |
| Local selection (initial, L1, L2) | · | · | · | · |
| Assembly levels 1–4 | · | · | · | · |
| Refinement | · | · | · | · |
| Painting | · | · | · | · |
| Founder refit | · | · | · | · |

Table: Accuracy of founder haplotypes through the pipeline on simulated crosses. [TO FILL from
`haplotypes evaluate` (founders.csv and founder_components.csv), pooled over contigs and
replicates, per scenario.]{.todo}

Phase was scored within released phase sets on correctly called true heterozygotes, separately for
samples with zero, one and two released parent edges, because phase in samples without released
parents rests on the children's chains or on convention (Table 2). Pedigree calls were compared
with the true parents among the sequenced samples (Table 3); a parent that was not sequenced is not
an observable edge, and an unresolved call is never counted as exact.

| Samples (released parent edges) | Genotype errors per called genotype | Phased share of true heterozygotes | Switch errors per scored site | Long switches | Released phase-set N50 |
|---|---|---|---|---|---|
| All | · | · | · | · | · |
| Two parents | · | · | · | · | · |
| One parent | · | · | · | · | · |
| No parent | · | · | · | · | · |

Table: Accuracy of released genotypes and phase on simulated crosses. [TO FILL from the phase
tables of `haplotypes evaluate`.]{.todo}

| Scenario | Exact parent configurations | Correct edges / true edges | Extra edges | Missing edges |
|---|---|---|---|---|
| [TO FILL]{.todo} | · | · | · | · |

Table: Accuracy of Tier-B pedigree calls on simulated crosses, against the true parents among the
sequenced samples. [TO FILL from the pedigree table of `haplotypes evaluate`.]{.todo}

[TO FILL: recombination: the fraction of called and of reported crossover intervals that contain a
true crossover, with their median widths, and true crossovers within the inferred informative
exposure against the expected count, on inferred edges that are true gametes.]{.todo}

The evaluation has no measure of per-sample local-ancestry (painting) accuracy or of relatedness
families against truth; painting is scored only through its relabelled founder panel, and families
only through the pedigree they inform.

## Analysis of the TroMau cichlid cross

The real dataset is a laboratory cross of Lake Malawi cichlids [@malinsky2018] between
*Astatotilapia calliptera* and a *Tropheops* species [TO FILL: species name]{.todo}. The input is a
single multi-sample BCF of biallelic SNPs with per-sample allele depths, restricted to the 22
autosomes [TO FILL: sample count, sequencing platform, mean and range of depth, number of SNPs per
contig, and how the BCF was produced]{.todo}. The sequenced samples include four individuals of the
founding generation of the cross, three *A. calliptera* females and one *Tropheops* male, together
with offspring sampled as first and second laboratory generations. These labels describe how the
animals were sampled; they were not used in the analysis, and no sequenced founding-generation
individual is assumed to be a parent of any other sample. The genetic map was a uniform
placeholder rate of 5 cM/Mb [TO FILL: confirm, or name the map used]{.todo}, so recombination
distances and map lengths below are in units of that placeholder unless stated otherwise. Because
that rate is not declared to be a measured genome-wide average, it provides no measured map length:
each chromosome counts as a single unit of map support in relatedness and in the pedigree's
release rules.

Because the real cross has no individual-level truth for its pedigree, we report internal
consistency, stability and explicit ambiguity rather than accuracy: candidate margins,
informative-site counts, support across chromosomes, stability under resampling, and the numbers
of missing-parent, ambiguous and unresolved outcomes.

::: todo
TO FILL from the 4.0 TroMau run of the validation matrix:

- per contig: sites kept by the site filter, samples flagged by the contamination filter, number
  of blocks, and the released founder count $K$;
- relatedness: number and sizes of full-sib families;
- pedigree, per release tier (A, B, C): released parent edges; children with two, one and no
  released parents; ambiguous and unresolved parent states; distribution of resampling support and
  of the number of supporting chromosomes; candidate margins;
- phase: phase-set lengths and the fraction of sites phased per sample;
- recombination: crossovers per meiosis and map length per chromosome.
:::

## Scaling with the number of founders and samples

Every stage has a work and storage model, evaluated at K = 30, and every stage was timed on
synthetic inputs over a ladder of founder counts (K from 5 to 30 at N = 200) and a ladder of sample
counts (N from 250 to 2,500 at K = 15), on one thread, after a discarded first call. The fitted
log–log slope of time against K or N, with its standard error, is shown for each of the 39 probed
scientific kernels in Figure 2 (Supplementary Table S1 lists them with what each probe covers).

![Measured scaling of the probed kernels of release 4.0. Each row is one kernel; points are fitted
log–log slopes of run time against K (left; K = 5–30 at N = 200) and against N (right; N =
250–2,500 at K = 15), with bars of one standard error. A K slope at or below 3.3 (dashed) is
accepted as at most cubic. Orange kernels belong to stages under a decided cost exemption, whose
worst case is bounded analytically rather than by the probed slope. Five operational kernels whose
size is only a proxy for K and N are omitted. Data: docs/developer/costs.md of the
repository.](figures/fig2_scaling.svg){#fig:scaling width=100%}

Every kernel gated as at most cubic in K had a measured K slope below the acceptance threshold of
3.3; the steepest were the posterior of the assembly site model ($3.06 \pm 0.11$), the family-phase
posterior ($2.81 \pm 0.02$), the painting transposition gains ($2.39 \pm 0.05$) and the pedigree
pair bound ($2.17 \pm 0.16$). The steepest kernel overall is the context matching of local
selection ($4.01 \pm 0.05$), which belongs to the rare context-coded screening covered by
exemption 0, whose analytic cost is $O(K^5)$ per round. Of the 37 stage rows of the cost table, 21
are at most cubic in their probed kernels, 11 fall under a decided exemption or exception, and 5
have no intrinsic dependence on K.

Four mechanisms have worst-case costs above cubic in K that the probed kernels do not bound:
structural screening in assembly joins, window fixing in refinement, label-continuity checks in
painting, and tie resolution in founder refit. They are carried as documented exceptions; none of
them binds on the validation inputs, and on those inputs founder refit used at most three rounds and
at most two tie members.

Against the number of samples, most kernels scale linearly. Those with slopes near two are the
stages that compare samples in pairs: the relatedness family screen ($2.00 \pm 0.00$), the optional
pairwise IBD output ($1.99 \pm 0.00$; off by default) and three pedigree kernels (pair bound,
exposure and family messages; $2.00$–$2.01$). Quadratic cost in N is intrinsic to a model in which
every sample is a candidate parent of every other.

[TO FILL: wall time, CPU time and peak memory of the whole-genome TroMau run and of a simulated
cross, on a stated node, from the 4.0 validation matrix.]{.todo}

## Reproducibility and agreement with an independent implementation

The released outputs of the implementation do not depend on the number of cores: the TroMau cross
gave identical outputs at 76 and at 56 cores. Release 4.0 is a reimplementation in which the
models of the previous release (3.5) were carried over rule by rule into a new execution engine,
and we used the earlier release as an independent check. A matrix of 55 runs compared the two on
simulated crosses with recorded truth, on the TroMau cross, on reference runs and on perturbed
inputs. At the time of release, 30 runs had finished, all with exit status 0, and 29 had
been scored [TO FILL: final counts when the matrix completes]{.todo}. In every scored run the founders,
paintings, founder refits, pedigree calls, phase and exported files matched the earlier release.
The only differences were declared in advance as numerical normalisations (Supplementary Table S2):
floating-point drift of at most $1.5 \times 10^{-9}$ in pedigree diagnostics, canonical numbering
of relatedness families, and diagnostic counts of rejected founder birth and death steps, which the
new implementation no longer repeats (lower by 1 to 28 per run). None of these changes a released
founder, painting, pedigree or phase call.
