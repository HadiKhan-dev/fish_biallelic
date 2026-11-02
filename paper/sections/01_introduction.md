# Introduction

Experimental crosses remain one of the most direct ways to connect genetic variation with
phenotype, to measure recombination and to study how divergent genomes interact in hybrids.
Their power rests on a simple structure: every chromosome of a descendant is a mosaic of a
limited number of founder haplotypes, joined at crossovers that occurred in the meioses of the
pedigree. Once the founder haplotypes and the mosaic of each individual are known, genotypes can
be completed from very sparse data, crossovers can be counted, and the pedigree itself can be
read off the shared segments.

Methods that exploit this structure have a long history. Hidden Markov models of inheritance
along a chromosome in a known pedigree [@lander1987], and of founder ancestry in multiparent
populations [@mott2000; @broman2019], reconstruct each individual's mosaic when the founder
genotypes and the design of the cross are known. Haplotype-copying models
[@li2003] describe a sample as a mosaic of other haplotypes and underlie most modern phasing and
imputation methods, including imputation from low-coverage sequence with large reference panels
[@rubinacci2021] and, without any reference panel, imputation from a small number of inferred
ancestral haplotypes [@davies2016]. Pedigree-aware phasing [@oconnell2014] and linkage mapping
from low-coverage data [@rastas2017] exploit known relationships. Pairwise relatedness can be
estimated directly from genotype likelihoods [@thompson1975; @korneliussen2015], and parentage and
sibship can be reconstructed from called SNP genotypes [@jones2010; @huisman2017].

Crosses in non-model organisms often fall between these assumptions. In the cichlid crosses that
motivated this work, drawn from the adaptive radiation of Lake Malawi [@malinsky2018], individuals
are sequenced at a few-fold coverage, so that genotype calls at single sites are unreliable. Not
every founder individual is sequenced, and those that are sequenced need not be parents of the
others. The number of distinct founder haplotypes is not known in advance and varies along the
genome, because founders may share haplotypes locally. Breeding records may be incomplete, and
cohort or generation labels describe when an animal was sampled rather than who its parents were.
A method that conditions on such labels, or on a declared design, risks reproducing errors in the
records as apparent biological signal.

Here we describe a method, released as version 4.0 of the software package
`haplotype_reconstruction`, that addresses this setting jointly. It reads only a multi-sample
VCF or BCF file with per-sample allele depths [@danecek2011; @danecek2021] and a genetic map;
sample names are treated as identifiers and nothing else about a sample enters inference. From
these inputs it

1. models read evidence and genotype likelihoods, with data-driven site and sample quality control;
2. estimates pairwise identity by descent and full-sib families from the genotype likelihoods of
   the whole genome;
3. discovers founder haplotypes in short marker blocks with a missing-aware model, and selects
   local founder panels using feedback from the chromosome-scale assembly;
4. assembles local panels into chromosome-scale founder haplotypes, deciding the number of founders
   with an explicit count rule and completing missing founder alleles jointly;
5. paints every sample as a diploid mosaic of the founders, and refits individual founder alleles
   with those paintings held fixed;
6. infers the pedigree, with every sequenced sample a candidate parent of every other, and releases
   parent calls in evidence tiers that keep missing, ambiguous and unresolved parents explicit;
7. refines sample phase conditional on the inferred pedigree; and
8. estimates recombination maps from the meiosis chains of the final phase.

Three principles run through the design. First, the inputs are the data alone: the direction of a
parent–offspring relationship comes from the genetic evidence, never from a cohort label or a sex
annotation, and the release rules prefer an explicit unresolved outcome to a forced call. Second,
the method is meant to remain usable as crosses grow: the documented scale envelope targets up to
N = 2,500 samples on bounded genome crops and K = 30 founder haplotypes, nothing is exponential in
K, and every stage is held to at most cubic work in K apart from a small number of documented
exemptions and exceptions. Third, results are
reproducible: all ties are resolved by declared rules, random draws come from named streams, and
the released outputs do not depend on the number of processor cores used.

We describe the models and decision rules of each stage, the simulation framework used to measure
accuracy against recorded truth, and the computational design, and we report the measured scaling
of every stage. [TO FILL: summary of the accuracy results on simulated crosses and of the analysis
of the TroMau cross.]{.todo}
