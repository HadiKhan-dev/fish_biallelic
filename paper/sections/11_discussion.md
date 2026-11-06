# Discussion

The method reconstructs founder haplotypes, pedigree, phase and recombination from allele depths
and a genetic map, with every sequenced sample considered as a candidate parent of every other.
The simulations show that this can recover observed parentage and accurate genotypes without
cohort labels or declared parental roles. They also show that success at one level does not
establish success at another. Locally accurate founder sequence can change true-founder identity
along a chromosome; accurate genotypes can coexist with an unresolved pedigree; and a high
phased share can coexist with shorter phase sets or long orientation errors.

Long-range founder identity is a substantial limitation in designs with unobserved parents and
founder haplotypes absent from parts of the sequenced descendants. For example, `std` had
30,360.1 one-to-one founder errors per million matched calls but 851.8 segment errors per million
calls, with 654 segment-label switches. Its truth included both absent and single-copy founder
alleles. The contrast is sharper in `longf3`: 64,503.9 one-to-one versus 132.7 segment errors per
million. A row that follows the correct local sequences can therefore be a mosaic of different
biological founders. Identity flags, withheld alleles and released segment boundaries qualify
that claim; they do not eliminate every wrong continuity assertion. Founder labels should not be
used as persistent biological identities across release segments, and good genotype accuracy
should not be cited as proof that founder identity is correct.

Chromosome support also limits pedigree release. The three-chromosome backcross recovered
0 of 120 observable edges and left all 80 states unresolved, whereas the eight-chromosome case
recovered all 120. This is an explicit abstention of the release rules, not a finding that parents
were absent. The map-support experiment recovered most edges on three chromosomes when an
explicit map supplied genetic length, but also changed the seed. It shows the importance of the
support definition without isolating a causal effect of chromosome count. An unmeasured fallback
rate should not be relabelled as a measured map merely to obtain more parent calls.

The $H=30$ test is a stronger warning against equating scale support with resolved biology. It
used five short contigs and 80 sequenced samples, 60 of whom had no observed parents. Only
24 parent states and eight exact configurations were correct. Exact calls recovered 4 of 40
observable edges; including supported marginal edges recovered 11. The panel could release the
true count on one chromosome while the pedigree remained weak. This design tests a particular
short-input, sparse-parent setting; it does not establish accuracy for arbitrary thirty-founder
crosses. Likewise, the combined read-stress design remained usable but had the lowest phased
share and most long switches in this matrix. Since depth variation, read error, imbalance,
dropout and partial observation changed together, these runs do not estimate separate effects or
a coverage threshold. Each design has only one seed, apart from the second seed of the small
cohort design; broader replication is needed for uncertainty across simulated datasets.

The separate $N=320$ series retains the same configuration and seed while changing mean depth
from 1 to 5. Founder wrong and missing alleles and genotype errors decreased over these four
observations, while all 600 observable parent edges and all 320 states were correct throughout.
Phase was not uniformly improved by depth: switch counts fell, but the phased share was lower
at depth 5 and the within-set site-error rate was not monotonic. One replicate per depth does
not establish a coverage threshold or uncertainty across crosses. These are depth comparisons
within 4.4, whose scientific rules and checked results retain 4.3's meaning.

TroMau supplies an application without individual-level parentage truth. The supported edges,
chromosome-resampling fractions and candidate margins describe inference, and the generation
labels cannot validate it. Chr3's component structure changed under tie-convention reversal.
That sensitivity limits claims of a unique founder reconstruction even when numerical-noise and
sample-order tests agree. Recombination output is further conditional on the inferred pedigree,
released phase and supplied fallback map. In long uninformative intervals its expected counts
approach the map model; absent exposure is not evidence of absent recombination. Reported
junction flags warrant inspection rather than automatic biological interpretation.

Search remains a methodological limit. Discovery's replacement starts and archive beam are
bounded, and local selection retains bank budgets, fit backstops, context-snap limits and capped
neighbourhood searches (cost exemption 0c). These can omit a better panel, even when every
admitted candidate is fitted correctly. Complete founder-count candidate admission does not make
the fixed-count search or the overall pipeline globally optimal. Shadow families, continuation
branches and refinement windows also have explicit limits; their unresolved outcomes and
unperformed searches should accompany downstream interpretation. Pedigree support is based on
a bounded candidate panel and composite evidence rather than enumeration of every joint
pedigree. Family-phase posteriors condition on the other samples' fitted configurations and are
not joint pedigree marginals.

The computational limits are equally explicit. Fitting every eligible count candidate costs
more at large $K$, and repeated deletion rounds have a quartic worst case under the quadratic
fixed-count work model. Rare context-coded selection is fifth order in $K$. Canonical refinement
windows, painting label-continuity checks and founder-refit ties retain the above-cubic worst
cases documented in the cost register; the cubic analytic bound for adaptive join screening
should not be confused with its direct-route query-count trend. Small measured kernel slopes
and small observed tie classes do not bound a whole adaptive search. Independently of $K$,
all-candidate relatedness and pedigree comparisons are quadratic in sample count. The documented
sample-scale envelope uses bounded genome crops and does not establish that any whole-genome
cross of that size will fit a particular machine. The 4.4 depth-2 analysis did complete on the
nominal 250 GB node that could not finish 4.3; depths 1 and 1.5 completed under configured
250 GiB limits on larger nodes. These limits and sampled pressure peaks do not estimate a
minimum memory requirement. Hardware, CPU counts and lane concurrency differ across the depth
series, so its memory variation cannot be assigned to depth, and its wall times do not establish
cross-node speedups.

Future work should redesign discovery and selection around a clearer search and stopping
contract, rather than increase fixed budgets. Promising directions are structural decomposition
of gauge ambiguity, bounds that certify which proposals can matter, and coordinated panel moves
that reach alternatives missed by local fits. The same redesign should preserve complete
founder-count candidate admission and explicit unresolved alternatives while reducing repeated
fits, window work and tie interactions. Evaluation should then separate local allele recovery,
founder identity continuity, supported parent edges and phase, over replicated crosses with
controlled chromosome support and parental observation. These are distinct scientific claims
and need distinct validation.
