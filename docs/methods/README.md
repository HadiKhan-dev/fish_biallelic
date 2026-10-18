# Methods

One chapter per pipeline stage, in the order the stages run. Each chapter states the stage's model, its
assumptions, thresholds and decision rules, how its outputs should be read, and what it costs. Internal model
constants are cited as `RootConfig.path = value` (for example `PaintingConfig.minimum_release_posterior = 0.9`);
[configuration.md](../configuration.md#internal-model-constants) lists all of them, and the documentation check
fails if a cited value differs from the code.

| stage | chapter | what it computes |
| --- | --- | --- |
| evidence | [evidence.md](evidence.md) | the record policy for VCF input, site and sample quality control, the read model fitted to the allele depths, genotype likelihoods |
| discovery | [discovery.md](discovery.md) | founder panels of each 200-site block from the pooled sample evidence, with missing alleles kept unknown |
| local selection | [local_selection.md](local_selection.md) | the local panel of each block under a diploid copying-path model, refined twice with candidates from context assemblies |
| assembly | [assembly.md](assembly.md) | completion of missing founder alleles, linking and joining panels into chromosome-scale founder components over the L1-L4 hierarchy |
| | [founder_refinement.md](founder_refinement.md) | converged refinement of the founder paths and founder counts after every final assembly level |
| painting | [painting.md](painting.md) | label continuity across atomic spans, then each sample as a diploid mosaic of the founders with an explicit unknown state |
| pedigree | [pedigree.md](pedigree.md) | observed-parent configurations from the paintings and genotype likelihoods, with direction, family evidence and support tiers |
| founder refit | [founder_refit.md](founder_refit.md) | founder alleles refitted from the painted carriers |
| family phase | [family_phase.md](family_phase.md) | pedigree-conditioned phase that preserves the called genotypes |
| recombination | [recombination.md](recombination.md) | conditional recombination maps, crossover intervals and observable meiosis exposure |
| export | [usage](../usage.md#export) | the final founder and sample VCFs, written from the records (no model) |
| all stages | [ties.md](ties.md) | the deterministic conventions that decide ties between equally supported alternatives (release 2.1; under review for 2.2), and what a run records about them |

## Conventions

- Alleles are 0 (reference), 1 (alternate) and -1 (unknown); a genotype counts alternate copies (0, 1, 2).
  Positions are VCF positions (1-based). Inside the records the run's samples are ordered by sample name, so no
  result depends on the input order; published tables and VCFs list samples in the input order (VCF header
  order, or the simulated observed individuals in pedigree order).
- Genotype likelihoods and the observed-read mask are separate. A site without reads has the flat likelihood
  (1/3, 1/3, 1/3) and is unobserved; a flat likelihood is never an observed heterozygote, and missing data never
  counts as an allele match.
- Founder haplotypes are local: a block panel, then a component of consecutive blocks after assembly. Founder labels
  are row indices within a component; the same row index in two components carries no identity. Components that
  the evidence does not connect stay separate and are never joined into one phase frame.
- H1/H2 (track) names in paintings and phase are arbitrary orientation labels, not maternal and paternal
  identities.
- The founder count K is decided from the data at every level; no biological founder count is supplied. The
  pipeline is designed for up to K = 30 founder haplotypes per block. Where a computation has an exact branch
  whose cost grows exponentially (exact site posteriors over at most ten free founders, exact maximum cuts up to
  K = 12), a polynomial method replaces it above the switch point, so no stage's cost grows exponentially in K; the
  cost section of each chapter gives the stage's complexity.
- One genetic map serves inference everywhere ([genetic_maps.md](../genetic_maps.md)): the switch and transition
  probabilities of local selection, assembly, painting, pedigree, family phase and recombination come from the
  contig's map, or from the fallback rate where there is none.
- Ties between alternatives the data do not separate (equal, or within one tie quantum) are decided by fixed
  conventions listed in [ties.md](ties.md); the run counts every such decision (`ties/` in the run directory).
- Every stage records its products with an identity (its configuration, upstream records and code), and nothing
  feeds back upstream: pedigree reads paintings, family phase reads the pedigree, recombination reads the phase.

## What the methods do not claim

Support tiers, bootstrap fractions and posterior thresholds measure internal stability and within-model
uncertainty, not calibrated probabilities of truth. Real crosses have no individual-level parentage truth: cohort
labels describe the design, not parentage, and no sequenced individual is assumed to be a parent unless the data
support it. Simulations with a known pedigree validate the methods ([validation.md](../validation.md)); they do
not establish accuracy on a particular real cross.
