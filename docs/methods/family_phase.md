# Family phase mechanisms

The complete semantic-unit descriptions are in [phase](./phase.md).

## edge rates

See [FP02](./phase.md#fp02) for the model, decision order, parameters and limits.

## Resumable component work

See [FP23](./phase.md#fp23) for durable solve, canonical sample-batch release and component units, identity binding and stage-owned cleanup. No checkpoint is taken inside a solve.

## Gauge-sample anchors and set witnesses

A sample with neither a linked parent nor children has no family frame, so each of its phase sets is anchored by a
gauge. Homologue 0 carries the label whose panel alleles at the anchor bin best match homologue 0 of the bin's most
probable label pair; ties keep that pair's labels at the ends of their score groups, then follow label index, and a
bin where the pair's alleles never differ uses label order. The set continues while the median orientation certainty
over each heterozygous bin's witness sites stays above `set_posterior`. A witness is a confidently heterozygous site
that the candidate label pairs, with their panel-allele flips, do not make confidently homozygous. A site
heterozygous only through the set statistic's site departure is not one. A sample without a linked parent but with
children uses the same statistic to decide whether to bridge its painting-tie breaks. See [FP18](./phase.md#fp18) and
[FP19](./phase.md#fp19).

## Recurrent wrong-allele reads

The site release fits a [painting-informed wrong-read rate](phase.md#site-specific-wrong-allele-reads)
on each contig before computing carrier support and release posteriors.

## Local phase abstention

A child can copy a parental homologue through a tract where both family solves
choose the wrong diplotype for the parent. Family phase propagates noncopy
support from observed ancestors and uses singleton tracks from the pre-refit
painting to check the local allele order. Within an existing phase set and an
untied painting frame, it withholds phase in high-support bins whose allele
order disagrees with the painting relative to both flanks. Genotypes and
phase-set joins are preserved. See [FP20](./phase.md#fp20).

## Read-class evidence

Family phase uses painting's effective foreign-read share and its full bin action
at confidence strength. The read factors use the selected sites, QC-corrected
genotype likelihoods and original observation flags. The component entry
uses the own law just before the first bin, or at site zero if the bin starts
there. The entry assigns foreign mass \(1-e^{-m}\) uniformly across donor
labels, where \(m\) is twice the share times the covering depth per homologue,
and zero for the libraries QC A corrected: their corrected likelihoods already
model their foreign reads ([painting](painting.md#foreign-read-class-and-bin-operators)).
Each site's P6 exclusion uses the two actual diplotype labels. Reversing the
diplotype swaps the own homologue groups and keeps each foreign donor's
identity.
One-class, deep and unshared rows keep their scalar scores exactly.

For each candidate pair, start the normalized incoming law \(a_{1,p}\) at the
component entry. At bin \(b\), apply the confidence-strength action to obtain
\(v = a_{b,p} M_{b,p}\). The emission is the scalar log likelihood plus the
bin log scale, the operator's output log gauge, and \(\log(\sum_j v_j)\).
The bin log scale includes painting's D5P correction from the same fixed-pair
carry. Where D5P applies, the read increment equals the confidence weight
times the untempered log predictive mass under the incoming law.
Carry \(a_{b+1,p} = v / \sum_j v_j\) to the next bin. The two homologue orders
get the same score since both the incoming law and the action are permuted
together. This scores each observation once along a constant pair. At a
label change the new pair uses its own hypothetical prefix. Both convention
solves, junction scoring, uncertainty and release use these emissions.
See [FP01](./phase.md#fp01) and the [painting read-state model](./painting.md#pr09).

The forward predictor applies the same formula at every share. At positive
share, a component compiles the extended bin action in float64 when one
strength's compiled actions of all its samples fit half the run's memory limit
and a common numerical gauge suffices; otherwise it applies the structured site
tape (both routes are exact).
With \(L\) sites per bin and at most \(K+1\) labels, compiling costs
\(O(LK^4)\) per sample and bin and each application \(O(K^2)\), against
\(O(LK)\) per application of the site tape. The compiled action is the faster
route while its storage, \(O(K^4)\) per sample and bin, leaves the samples'
concurrency intact; the memory rule hands larger panels to the walk, so the
asymptotic work stays cubic. At zero share the
fixed own-class width gives \(O(LK^2)\) work including the bin products.
