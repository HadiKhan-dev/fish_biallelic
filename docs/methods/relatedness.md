# Relatedness

Release 3.2. The relatedness stage estimates how the samples are related from their genotype likelihoods alone: the
identity by descent (IBD) of every pair along the genome, and a partition of the samples into full-sib families with
latent or sequenced parents. It reads only the evidence records: no founder panel, painting, pedigree, phase or cohort
label. No later stage feeds back into it. In 3.2 its only consumer is the pedigree release's
[sibling-as-parent guard](pedigree.md#sibling-as-parent-guard-release-32).

The products are inferences from genotype likelihoods, not pedigree truth. A sample's family is a hypothesis with a
measured support across map units (chromosomes without a genetic map), and the stage keeps unsupported placements as
unresolved rather than forcing a call.

## Inputs and outputs

- Input: each contig's evidence record (genotype likelihoods, observed-read mask, positions) and the run's genetic
  maps (the IBD chain's switching distance).
- Run record `relatedness` (`relatedness.records.RelatednessProduct`), bound to every contig's evidence identity,
  each contig's map and the stage settings.
- Published tables in `OUTPUT/relatedness/`:
  - `families.csv`: one row per sample. Its family, the share of support units on which that family is its best
    placement (`FamilySupport`), and the family's two parents with their support. A parent is a sequenced
    sample or empty for a latent parent.
  - `pairs.csv`: one row per sample pair. The posterior IBD0, IBD1 and IBD2 shares of the pair's jointly observed
    sites, the number of those sites, and the IBD0 and IBD2 stretch evidence.
  - `half_sib_links.csv`: pairs of latent-parent families better explained as sharing one latent parent.
  - `summary.json`: the family fit (objective, sweeps, convergence, sites per chromosome).

## Allele frequencies and sites

Each contig's alternative allele frequency p is fitted by EM under Hardy-Weinberg over all samples' genotype
likelihoods (six iterations; unobserved cells carry the prior), clipped to [1e-4, 1 - 1e-4].

- The IBD chain uses every `RelatednessConfig.ibd_site_step = 10`th site of the contig, at any frequency.
- The family model uses every `RelatednessConfig.family_site_step = 20`th site among those with a minor allele
  frequency of at least `RelatednessConfig.minor_allele_frequency = 0.05` and reads in at least
  `RelatednessConfig.minimum_observed = 0.5` of the samples.

## Pairwise identity by descent

For each pair (a, b), a three-state chain (IBD0, IBD1, IBD2) runs over the contig's IBD sites. It leaves its state at
`RelatednessConfig.ibd_switch_rate = 4.0` per Morgan, to either other state equally; the genetic distance comes from
the run's map. A site observed in both samples emits, with Hardy-Weinberg genotype probabilities h at frequency p
(q = 1 - p) and the samples' genotype likelihoods A and B:

    IBD0: (sum_g h_g A_g) (sum_g h_g B_g)
    IBD2: sum_g h_g A_g B_g
    IBD1: q^3 A0 B0 + p^3 A2 B2 + p q A1 B1 + p q^2 (A0 B1 + A1 B0) + p^2 q (A2 B1 + A1 B2)

Other sites emit 1. Genotype likelihoods need no normalisation, because every state of a pair uses the same reads.

Outputs, summed over the contigs:
- the posterior state shares over jointly observed sites (forward-backward);
- `lr_ibd0` = log L(IBD0, IBD1, IBD2) - log L(IBD1, IBD2): the evidence for stretches with no IBD. A parent and its
  child have none and full sibs share about a quarter of the genome IBD0. In crosses from few founders, with
  low-depth genotype likelihoods, the model still finds some IBD0 evidence between parent and child: on the simulated
  designs their median is 20-35% of the full sibs' median. Consumers should compare `lr_ibd0` values within one run,
  as the guard does with a ratio, rather than use it as an absolute scale;
- `lr_ibd2` = log L(IBD0, IBD1, IBD2) - log L(IBD0, IBD1): the evidence for stretches IBD on both homologues.

The restricted chains move only among their allowed states.

## Full-sib families

### Model

A family is a set of samples (full sibs) with two parent slots. Each slot is either latent (Hardy-Weinberg at the
sample allele frequency) or a sequenced sample p. Its composite log-likelihood over the family sites is

    L_F = sum_i log sum_{g_u, g_v} P_u(i, g_u) P_v(i, g_v) prod_{c in F} sum_g T(g | g_u, g_v) GL_c(i, g),

- T is Mendelian transmission.
- GL_c is the member's genotype likelihood, scaled to a maximum of 1 per site. Unobserved sites contribute 1.
- P is the Hardy-Weinberg prior for a latent slot, or the sequenced parent's genotype posterior from its own reads.

The objective is the sum over families.

**Each sample's reads are counted once.** They enter the likelihood as a member of the sample's own family. A family
with a sequenced parent conditions on that parent's reads, through the posterior: the factor
P(children | parent's reads). The sum is therefore an approximate chain-rule factorisation of the joint likelihood,
which is valid only along an acyclic parent graph. Every assignment is checked, and none may make a sample its own
ancestor. The approximation drops what a sequenced parent's own family says about its genotype, and treats two families
that share a sequenced parent as independent given that parent's reads.

**Half-sib links.** Two latent families may instead share one latent parent, their other parents being independent:

    L_HS(i) = sum_{g_u} pi(g_u) [sum_{g_v} pi(g_v) prod_F ...] [sum_{g_w} pi(g_w) prod_G ...].

A full-sib merge is made only when it beats both separation and this link. A link that beats separation by more than
log N is recorded, and the families stay apart.

### Search

The search is deterministic.

1. **Agglomeration.** Starting from one-member latent families, repeatedly merge the pair with the largest positive
   full-sib gain that also beats its half-sib link.
2. **Sweeps**, at most `RelatednessConfig.max_sweeps = 30`, until a sweep changes nothing:
   1. *Parent assignment.* At most one assignment per family with a free slot, best offers first, applied when the
      gain exceeds log N (the choice among N samples) and the parent graph stays acyclic. An offer is either a
      sequenced non-member, or a member moved first to its best other placement.
   2. *Moves.* Each sample moves to its best valid placement: another family, or a new one-member family.
   3. *Merge.* The single best merge of two families with compatible parents: at most two sequenced parents, none of
      them a member.

### Support

Support is measured on each support unit alone, against the fitted partition. A chromosome with a supplied
genetic map is cut into max(1, floor(L / `RelatednessConfig.support_unit_morgans = 1.0` Morgan)) consecutive units of
equal measured map length. L is the measured map length of its family-site span: the part between the map's first
and last knots, or all of it when `[genetic_map] rate_is_measured = true` (`RelatednessConfig.rate_is_measured =
False` by default). Sites beyond the knots join the end units. A chromosome with neither a map nor a stated rate is
one unit. This is the pedigree's [map-length rule](pedigree.md#map-length-requirement-release-32).

- A member's family is its best placement on that unit when
  log P(member's reads | the rest of its family) beats every other family and being alone.
- A sequenced parent beats a latent one on that unit when the family's likelihood with it is higher than with a
  latent parent in its slot.

`member_support` and `parent_support` are the shares of units on which these hold. They are member-removal contrasts
on the partition fitted to all the data, which the member and the unit helped choose. They are not held-out
validation, and not calibrated probabilities. A consumer decides the support it requires. The guard uses members whose family is their best placement on at least 75% of the support units
(`PedigreeConfig.guard.member_support = 0.75`).

**Held-out chromosomes.** The families were refitted without each chromosome in turn, then tested on that
chromosome alone. The share of members whose refitted family is their best placement there:

| run | members with support ≥ 0.75 | other members | full-sib pairs kept together |
| --- | --- | --- | --- |
| TroMau-like | 0.92 | 0.42 | 98% |
| eight-chromosome F3 | 0.84 | 0.42 | 97% |
| half-sib/backcross control | 0.85 | 0.44 | 85% |
| eight-chromosome backcross | 0.73 | 0.39 | 60% |
| real TroMau (22 chromosomes) | 0.95 | 0.07 | 94% |

Support predicts what held-out chromosomes confirm, but it is not their probability.

**Half-sib links** are pairwise hypotheses, not a consistent graph of shared latent parents. The half-sib comparison
guards the agglomeration and the merges, but not single-sample moves, so the search does not optimise jointly over
shared latent parents. Every sample receives a family; consumers abstain through the support they require.

## Interpretation and limitations

- **Hardy-Weinberg at the sample frequency.** No inbreeding or background-relatedness term is fitted. In a cross
  from few founders most pairs are related, so the model separates large full-sib families well but resolves small
  families in parent-rich designs poorly.
- **Measured on saved simulations** (full-sib pair precision and recall over all members; then over members with
  support at least 0.75):

  | design | all members: precision | all members: recall | supported members: precision |
  | --- | --- | --- | --- |
  | TroMau-like | 0.975 | 0.953 | 1.000 |
  | F3, eight chromosomes | 1.000 | 0.961 | 1.000 |
  | backcross, eight chromosomes | 0.13 | 0.09 | 0.57 |
  | standard | 0.00 | 0.00 | no supported pair |
  | standard, held-out seed | 0.11 | 0.05 | no supported pair |

  In the small-family designs few members reach the support threshold, so consumers abstain there.
- **Composite likelihood.** The objective sums over linked sites, so it overstates confidence. Per-unit support,
  not the objective's margin, is the stability measure.
- **Sequenced-parent slots** are often wrong in small families; for example, 46 of 65 are right on the backcross
  design. The guard does not use them.
- **No cohort labels.** The stage uses no cohort, sex or generation label, and no assumed parentage. Real-data
  families are hypotheses, never truth.
- **Placement.** The stage depends only on the evidence. It runs at the start of the genome-wide stages, after every
  contig is painted, in every execution mode. A consumer before assembly (pedigree-aware founder identity, planned)
  would need it to move ahead of assembly, which serial mode does not yet allow: there the evidence of later contigs
  is not yet prepared.

## Cost

- **Time.** The IBD chain costs O(N^2 L / step) for N samples and L sites. Each search sweep costs O(N F M) family
  evaluations, for F families and M family sites. Nothing depends on the founder count K.
- **Parallelism.** Pairs, families and candidate parents run as Numba parallel loops over the stage's CPU budget.
- **Memory.** The child terms and the family sums hold O(N M) float64 values.
- **Measured.**

  | run | cores | wall time | peak resident memory |
  | --- | --- | --- | --- |
  | real TroMau (116 samples, 22 chromosomes, 95,412 family sites, 6,670 pairs) | 76 | 162 s | 3.4 GB |
  | simulated designs (40-115 samples, 3-8 chromosomes) | 38 | 44-142 s | under 3 GB |
