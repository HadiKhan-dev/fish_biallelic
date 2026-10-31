# Relatedness

Release 3.2. The relatedness stage estimates how the samples are related from their genotype likelihoods alone: the
identity by descent (IBD) of every pair along the genome, and a partition of the samples into full-sib families with
latent or sequenced parents. It reads only the evidence records: no founder panel, painting, pedigree, phase or cohort
label. No later stage feeds back into it. Its only consumer is the pedigree release's
[sibling-as-parent guard](pedigree.md#sibling-as-parent-guard), which reads the full-sib families (since release 3.4,
not the pairwise IBD statistics).

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
  - `summary.json`: the family fit (objective, sweeps, convergence, the search kept and its margin, sites per
    chromosome).

## Allele frequencies and sites

Each contig's alternative allele frequency p is fitted by EM under Hardy-Weinberg over all samples' genotype
likelihoods (six iterations; unobserved cells carry the prior), clipped to [1e-4, 1 - 1e-4]. Since release 3.4 the
EM runs per site in one parallel pass, a mathematically equivalent form of the earlier array expression: it agrees
bitwise on every validation replay checked, and other inputs can differ in the last bit (the sums over samples may
associate differently), which leaves the site selection and every decision unchanged in the checks. With two or more
CPUs the next contig's evidence is read and decoded on one CPU while the current one is computed on the others.

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
  designs their median is 20-35% of the full sibs' median, and the posterior IBD0 share of true parent-child pairs
  has a median of 0.38. The absolute scale is therefore not interpretable in such crosses: the Hardy-Weinberg null at
  the sample frequency treats founder haplotypes shared by chance as unrelated. Consumers should at most compare
  `lr_ibd0` values within one run (the 3.2-3.3 sibling guard used a within-family ratio; release 3.4 replaced it by a
  Mendelian family-consistency test);
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

The search is deterministic, and it does not depend on sample names or on the input order (release 3.4). The
samples' order on the run's axis follows their names, so any choice made in that order would make the families depend
on what the samples are called.

1. **Agglomeration.** Starting from one-member latent families, repeatedly merge the pair with the largest positive
   full-sib gain that also beats its half-sib link.
2. **Sweeps**, at most `RelatednessConfig.max_sweeps = 30`, until a sweep changes nothing. They run twice from the
   agglomeration, once with each kind of move (step 3):
   1. *Parent assignment.* At most one assignment per family with a free slot, best offers first, applied when the
      gain exceeds log N (the choice among N samples) and the parent graph stays acyclic. An offer is either a
      sequenced non-member, or a member moved first to its best other placement. Offers are ranked on the partition
      at the start of the sweep, and each is scored again just before it is applied.
   2. *Moves.* A sample moves to its best valid placement, another family or a new one-member family, when that
      gains more than 1e-6. A target is valid when none of its sequenced parents is the sample or descends from it.
   3. *Merge.* The single best merge of two families with compatible parents: at most two sequenced parents, none of
      them a member.
3. **Two searches; the better one is kept (release 3.4).**
   - *Best improvement.* Repeatedly, the single move with the largest gain over all samples is applied. This stops
     when no move gains more than 1e-6, or after N moves (the next sweep continues).
   - *Content order.* Each sample is visited once per sweep, by decreasing content key (below), and moves at once
     when its best placement gains more than 1e-6.
   - The fit with the higher objective is kept. Within 1e-6, the best-improvement fit is kept. `summary.json` records
     the search kept (`search`: `best_improvement` or `content_order`) and its objective minus the other's
     (`search_margin`).
   - The objective has many local optima, 10^2 to 10^4 log units apart on the simulated designs, and neither search
     reaches the better one on every design. Keeping the better of the two never fits worse than either alone.
   - Until 3.3, a single first-improvement search visited the samples in axis order (name order), so renaming the
     samples could change the partition (on a real three-chromosome TroMau subset: 35 against 36 families).

**Ties.** Numerical gains and evidence keys are separate. Among eligible choices within 1e-6 of the
largest gain, choose the largest evidence key. Compute each sample's key once from its full retained child-term
vector: the log probabilities of its own normalised genotype likelihoods under all nine parental genotype pairs,
with unobserved cells contributing zero. These terms do not use the cohort allele frequency. Compare the full
vectors lexicographically by numeric value in site/parent-genotype order, stopping at the first difference;
no rounding or gain tolerance enters this exact
evidence comparison. Dense ranks preserve the order for later choices. Unlike digest ordering, numerical
lexicographic ordering does not scramble unrelated samples because an upstream fit changes a last bit.
A family key contains the complete sorted member-rank multiset and its sorted parent ranks (latent = -1); a merge
key is the sorted pair of family keys. Use these keys for agglomeration, move targets (including a new singleton),
member out-placement, parent offers, family merges and the content-order sweep. Member sums on restoration also
follow evidence order. Sample names and input indices never distinguish evidence.

Exactly identical retained evidence vectors share a key and are interchangeable: output identities within that
class are defined only up to permutation. No evidence-only method can identify which of two identical samples
occupies a nonsymmetric role. For distinct evidence vectors, even equal marginal log-likelihoods have distinct
keys. An isolated parent-offspring direction can remain unidentified by the likelihood; its evidence-key choice
is arbitrary and reproducible, not biological evidence for that direction.

**Incremental gains.** The best-improvement moves keep every sample's gain on joining every live family, and on
leaving its own.
A move changes only its two families, so only their gains are scored again: every sample joining either of them, and
their members leaving. The selected move is scored once more on the current partition before it is applied, so every
applied move raises the objective by more than 1e-6, and the search cannot cycle.

**Implementation (release 3.4).** The search above is computed without changing any value it compares or any
decision it makes; only which gains are evaluated, and when, changed.
- *Kept values.* Every family carries a version that each change of its members, sums or slots advances. A sample's
  join and leave gains, a family's parent offers and a pair's merge gains are kept for the versions they were computed
  for, across moves, parent assignment, merges and both searches, and are evaluated again only once those change.
- *Screens.* Most gains are far from any decision: an unrelated pair is never a merge candidate, and most
  placements and parents are far below a sample's or a family's best. The full-sib and half-sib merge gains of the
  agglomeration, every join gain and every parent-slot offer are first estimated in the linear domain: each family's
  and sample's per-site terms, rescaled to their largest term and stored in float32, so that a site's sum of
  exponentials is a short dot product (a site whose sum is too small for float32 is evaluated in the log domain).
  A candidate is set aside only when its estimate is beyond the decision by more than an error bound with two
  parts: the float32 terms (about 1.2e-7 log units per site, nonnegative sums without cancellation, covered by
  1e-6 per site) and the float64 reductions of the estimate and of the exact gain (at most about (M + 16) 2^-53
  times the summed magnitudes of the log-likelihoods involved, for each, covered by twice that), plus 0.1 log units.
  This is a first-order rounding bound; measured screen errors are far below it (at most 2.7e-5 log units over
  56,502 sites on the validation designs). The candidates set aside are an agglomeration pair whose full-sib gain is
  negative, a half-sib comparison already decided, and a placement or parent that can neither be the best nor tie
  with it. Everything else is evaluated exactly, with the same expression as the previous implementation, so the
  merges, moves, placements, parents and ties are those of the exact gains: every validation replay agrees with the
  previous implementation bitwise.
- The largest agglomeration gain comes from per-row maxima instead of a scan of the gain matrix, and support and the
  half-sib links are evaluated in single parallel kernels.
- `relatedness.families.fit(..., reference=True)` runs the previous implementation (`families_reference`),
  for validation.

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
- A margin within 1e-6 is a tie, not a win (release 3.4). A latent one-member family is exactly its member alone, and
  a sequenced parent without reads on a unit is exactly a latent one. Until 3.3, float noise decided these
  comparisons, and that noise follows the sample order. A latent one-member family now always has support 0.

`member_support` and `parent_support` are the shares of units on which these hold. They are member-removal contrasts on
the partition fitted to all the data, which the member and the unit helped choose. They are not held-out validation, and
not calibrated probabilities. A consumer decides the support it requires. The guard uses members whose family is their
best placement on at least 75% of the support units (`PedigreeConfig.guard.member_support = 0.75`).

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
- **Placement (release 3.3).** The stage depends only on the evidence. It runs once every contig's evidence is
  recorded and before any assembly, in every execution mode: after discovery in the default mode, after an
  evidence-only pass over every contig in serial mode, and before any shard (which needs the product recorded by a
  run with `[run] stop_after = "relatedness"`). Nothing downstream feeds back into it.

## Cost

- **Time.** The IBD chain costs O(N^2 L / step) for N samples and L sites, evaluated exactly for every pair; it is
  the largest part at large N. The agglomeration screens all O(N^2) sample pairs (O(N^2 M) float32 multiply-adds for
  M family sites, about a tenth of an exact evaluation's cost) and evaluates exactly only the pairs whose full-sib
  gain may be positive. Each applied move re-estimates its two families' join gains for every sample (O(N M) screen
  terms) and evaluates exactly only the placements near the best; parent offers and merges are kept per family
  version. The sweeps run twice (two searches). Nothing depends on the founder count K.
- **Parallelism.** Pairs, families, samples, candidate parents and a family's sites run as Numba parallel loops over
  the stage's CPU budget; a family's log-likelihood sums its sites' terms in site order. While the next contig's
  evidence is read (one helper thread, decoding on one CPU), the kernels run on the budget less one CPU.
- **Memory (release 3.4, exact storage reduction).** Every unordered pair is still evaluated; no candidate set,
  likelihood, threshold, or search rule changes. Agglomeration stores its upper triangle in `N(N-1)/2` float64
  cells (0.40 GB at N = 10,000), replacing the `(2N+2)^2` gain table and separate `N^2` screen. Screen estimates
  are overwritten by exact gains in batches of 262,144 pairs before any merge is selected. At most N family slots
  are needed: every live family contains a sample. Singleton sums borrow their child terms; changed families own
  their sum rows, released rows are freed, and the saved search start shares rows until a mutation copies them.
  Restoring a sum slices the site axis into at most 32 MiB of input terms, retaining member reduction order.
  The saved start keeps gain-cache rows only for live families.
- The child terms, genotype posteriors, and RELFAST screens remain O(N M); gain caches and released IBD summaries
  remain O(N^2). The screens add about 64 N M bytes during agglomeration and 56 N M during the searches. IBD pair
  indices are filled directly in their final buffer. Its four accumulated posterior columns are normalised in
  place in bounded batches, and the released means and site counts are contiguous views of that buffer. The
  evidence prefetch still holds up to two decoded contigs. This reduces resident memory; it does not make the
  all-pairs computation subquadratic or bound whole-contig evidence by a site window.
- **Measured** (one 76-core node; simulated runs of the N-scaling series; before -> after the speed-up).

  | run | evidence pass (IBD and terms) | family fit | peak resident memory |
  | --- | --- | --- | --- |
  | N = 1,003, 8 chromosomes, 58,425 family sites | 952 s -> 252 s | 1,416 s -> 82 s | 27 GB -> 20 GB (evidence pass) |
  | N = 2,503, the same chromosomes cut to 10% | 319 s -> 126 s | 2,391 s -> 65 s | |
  | N = 1,003, cut to 10% | 97 s -> 24 s | 214 s -> 12 s | |

  At N = 2,500 over the same 8 full-length chromosomes, scaling each part by the cut series, the stage takes about
  29 min (22 min evidence pass, 7 min family fit), against about 4.9 h before the speed-up.

The pedigree guard's cleaning uses the final retained edges, with irreversible withdrawals until the retained set is
stable. This removes stale cleaning premises, but family support is placement stability, not a guarantee that most
witnesses are offspring of the tested parent. An unsequenced shared co-parent, uneven mating sizes, or incomplete or
wrong released-parent markers can cause true-edge withdrawals. Absence of a released parent edge is not evidence
that the biological parent is absent. See the [guard's assumptions and diagnostics](pedigree.md#sibling-as-parent-guard).
