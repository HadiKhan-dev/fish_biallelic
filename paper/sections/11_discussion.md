# Discussion

We have described a method that reconstructs the founder haplotypes, pedigree, phase and
recombination of an experimental cross from low-coverage sequencing, using only the allele depths
of the sequenced samples and a genetic map. Its central choice is to let the data alone determine
who is related to whom: no cohort, generation, sex or breeding record enters inference, every
sequenced sample is a candidate parent of every other, and the direction of each relationship is
inferred. The cost of that choice is that the method cannot be told which samples are founders or
parents; the benefit is that errors in records cannot become apparent biological signal, and that
the same pipeline applies to crosses whose records are incomplete or absent.

[TO FILL: two or three paragraphs interpreting the simulation and TroMau results: where founder
reconstruction, painting, pedigree and phase are accurate, how accuracy degrades with depth,
missingness, backcrossing and unsequenced parents, and what the TroMau pedigree shows.]{.todo}

Several approximations should be kept in mind when interpreting the outputs. Discovery and assembly
fit finite candidate panels with declared search orders and budgets, so a founder haplotype that no
search reaches cannot be reported. Founder rows are inferred haplotypes within an assembled
component; their labels are not chromosome-wide biological identities, and breaks, continuation
ambiguity and identity flags qualify comparisons across components. The pedigree is chosen from a
bounded candidate panel with composite evidence and chromosome-level calibration; its support
fractions measure stability under resampling, not the enumeration of all joint pedigrees. Family
phase releases modular calls conditional on the inferred pedigree rather than a single globally
identified assignment of haplotypes. Recombination maps are conditional on the fixed inferred
pedigree and the final phase, a supplied genetic map is a model input rather than observed truth,
and an uninformative interval is not evidence for the absence of recombination. Because the method
cannot exclude samples by label, individuals that do not belong to the cross must be removed from
the input before analysis.

On the computational side, every stage is designed to be polynomial in the number of founder
haplotypes, and the measured kernels of every stage gated as at most cubic in K stayed below that
bound up to K = 30. Two kinds of cost remain outside that bound. Four decided exemptions keep
bounded searches whose cost is documented and measured: rare context-coded screening in local
selection, a fixed cap of nine ranked refits in assembly and refinement, bounded discovery and
selection searches, and the optional pairwise IBD output. Four mechanisms have worst-case costs above cubic
that are carried exactly as documented exceptions: structural screening in assembly joins, window
fixing in refinement, label-continuity checks in painting and tie resolution in founder refit.
None of them binds on the inputs analysed here, but they define where further algorithmic work is
needed before crosses near K = 30 are analysed routinely; they are scheduled for review, as is the
cap on ranked refits, which does not grow to enumerate every candidate at larger K. Independently
of K, the pedigree is quadratic in the number of samples, because every sample is compared with
every other as a candidate parent. The documented scale envelope of N = 2,500 samples was
established on bounded genome crops; it does not by itself establish that every whole-genome
dataset will fit a given machine.

Finally, the implementation treats reproducibility as part of the method. Ties between numerically
indistinguishable alternatives are resolved by declared content-based rules, random draws come from
named streams with recorded seeds, sums are taken in declared orders, and released outputs do not
depend on the number of cores. These properties made it possible to compare a complete
reimplementation of the pipeline with its predecessor run by run, and to attribute every observed
difference to a declared numerical normalisation.
