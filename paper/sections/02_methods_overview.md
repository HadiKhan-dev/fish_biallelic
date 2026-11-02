# Methods

## Overview and notation

The method is a single pipeline of stages connected by a fixed dependency graph (Figure 1).
Stages that act within one chromosome (contig) run on each contig independently and concurrently;
two stages, relatedness and pedigree, pool information across the genome. Relatedness consumes the
evidence of every contig and nothing else, so that families are estimated from genotype likelihoods
alone, without founders, painting, pedigree, phase or labels; it runs alongside the per-contig
reconstruction, which does not depend on it. The pedigree consumes the
relatedness families together with the paintings and genotype likelihoods of all contigs. Family
phase and recombination act per contig after the pedigree is fixed, and never feed back into
founder or pedigree inference. A genetic map, in centimorgans, gives the genetic distances used by
relatedness and by every stage from local selection onwards; where no map is given, or beyond its
last position, a uniform fallback rate is used. Founder discovery works within short blocks and uses
no map.

![The stage graph of release 4.0. Each per-contig chain (blue) runs independently for every
contig. Relatedness consumes only the evidence of all contigs and runs alongside the per-contig
reconstruction; pedigree inference pools the relatedness families, paintings and genotype likelihoods of all contigs
(orange). Local selection proceeds in three rounds (initial, L1 and L2); the second and third use
an assembly context built from the preceding round. Family phase and recombination are conditional
on the inferred pedigree and do not feed back upstream. The genetic map enters relatedness and every
stage from local selection onwards.](figures/fig1_pipeline.svg){#fig:pipeline width=100%}

Throughout, $N$ is the number of sequenced samples, indexed by $i$; $K$ the number of founder
haplotypes, indexed by $k$; and $L$ the number of biallelic sites on a contig, indexed by $j$, of
which $M$ may be selected for a given computation. $B$ denotes the number of bins or blocks into
which a contig is divided. Painting, pedigree and phase models use $S \le K + 1$ states per
haplotype: one per class of founder rows that are identical over the painted sites, and one
background state. For
sample $i$ at site $j$, $r_{ij}$ and $a_{ij}$ are the reference and alternate read counts and
$d_{ij} = r_{ij} + a_{ij}$ the depth; $g \in \{0, 1, 2\}$ is the alternate-allele dosage of a
diploid genotype; and $h_{kj} \in \{0, 1\}$ is the allele carried by founder $k$. Where a stage needs the
probability of recombination in a meiosis over a genetic distance of $x$ morgans, it uses the
Haldane recombination fraction
$$r_H(x) = \tfrac{1}{2}\left(1 - e^{-2x}\right) = -\tfrac{1}{2}\,\mathrm{expm1}(-2x),$$
evaluated in this form for numerical accuracy at short distances [@haldane1919]. The copying models
of local selection, assembly and refinement instead convert map distance into switch probabilities
of their own, described with each stage. Further symbols are defined where they are used.

**No sample metadata.** Sample names identify rows of the input and carry no other meaning. No
generation, cohort, sex, breeding record or known-parent label enters any stage. Every sequenced
sample is a candidate parent of every other, parent slots are unordered and carry no sex, and the
direction of a parent–offspring relationship is inferred from the data. Individuals that are not
part of the cross must be removed from the input, because the method cannot exclude them by label.

**Uncertainty is released, not hidden.** Founder alleles that the data do not determine remain
missing rather than imputed; painting, pedigree and phase calls carry their support; and parents
that were not sequenced, are ambiguous between candidates, or cannot be resolved are reported as
such. Every search in the pipeline has a declared order and budget, and every tie between
numerically indistinguishable alternatives is resolved by a declared rule (Implementation, below).

The sections below give the model and decision rules of each stage. The Supplementary Information
(sections S1–S6) gives their finer rules with the default value of every named parameter, and
codes there such as EC05 or AJ17 point to the corresponding unit of the software's methods
documentation.
