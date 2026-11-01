# Founder allele refit

Refit uses released painting labels to reconsider individual founder alleles with untempered site evidence. Labels are held fixed: this is not another ancestry search. N is scoring samples, L sites and K founder rows. Tied incompatible updates release unknown alleles. Pedigree consumes the frozen pre-refit painting and does not receive this output as feedback.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="fr01"></a>
## fr01 Refit scoring set and founder-pair table (released labels expanded to sites, founder view, untempered robust logs)

Expand released painting labels onto sites and form untempered robust genotype evidence from the founder view. Score only samples admitted by the refit rule. Sites with no scoring sample retain their alleles; painting’s tempered bin objective is not reused as the site objective.

Implementation: [science/refit/refit.py](../../haplotype_reconstruction/science/refit/refit.py), [science/painting/release.py](../../haplotype_reconstruction/science/painting/release.py).

Parameters by name: `robustness_epsilon`, `REFIT_TIE_QUANTUM`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fr02"></a>
## fr02 Refit coordinate ascent: flip and two-row phases, rounds, interaction, sweeps, hysteresis gains

Within each site, run the named sweep schedule with flip rounds before two-row rounds. Rank qualified moves, withhold conflicting tied alternatives, otherwise apply independent moves and rescore affected candidates. Limit each row’s changes as declared. Use the single direct two-row gain for qualification and recheck (L10).

Implementation: [science/refit/refit.py](../../haplotype_reconstruction/science/refit/refit.py).

Parameters by name: `minimum_gain_nats`, `coordinate_sweeps`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K^2); O(P); O(n log n). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fr03"></a>
## fr03 Refit tie classes and decision order (carrier-scaled quanta, wide classes, convention)

Sort exact gains stably, form carrier-scaled tie components and identify wide classes. Ordinary classes use the row/pair convention; wide classes retain value order as distinct choices. Reversal changes order within allowed classes, never their membership.

Implementation: [science/refit/refit.py](../../haplotype_reconstruction/science/refit/refit.py).

Parameters by name: `REFIT_TIE_QUANTUM`, `REFIT_TIE_QUANTUM_PER_CARRIER`, `WIDE_QUANTA`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n log n). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fr04"></a>
## fr04 Refit ambiguity release (tied mutually exclusive moves released unknown)

For mutually interacting moves in one tie class, apply each hypothetically and test whether the other still qualifies. Mark every conflicting pair in one pass and release all involved alleles as unknown before rescoring. A called allele withheld by this ambiguity cannot be filled later in the site.

Implementation: [science/refit/refit.py](../../haplotype_reconstruction/science/refit/refit.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(K); O(K^4); O(K^5). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fr05"></a>
## fr05 Refit fill rule for alleles the painted panel left unknown

After the coordinate sweeps, visit alleles that were already unknown in the painted panel. Test the declared allele order, minimum gain and supporter gates inclusively. Preserve unknown on insufficient evidence; this fill cannot undo ambiguity withholding of an originally called allele.

Implementation: [science/refit/refit.py](../../haplotype_reconstruction/science/refit/refit.py).

Parameters by name: `minimum_gain_nats`, `fill_min_supporters`, `REFIT_TIE_QUANTUM`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fr06"></a>
## fr06 Refit stage, product, records and downstream reach

Publish refit component panels and diagnostics under evidence/painting/parameter identities. Resume only compatible records. Evaluation reads the original refit axes; export consumes the separate release/withholding view. Pedigree continues to use its pre-refit founders.

Implementation: [stages/refit/tasks.py](../../haplotype_reconstruction/stages/refit/tasks.py), [science/refit/run.py](../../haplotype_reconstruction/science/refit/run.py), [stages/refit/records.py](../../haplotype_reconstruction/stages/refit/records.py), [contracts/refit.py](../../haplotype_reconstruction/contracts/refit.py).

Parameters by name: `SITE_TILE`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Per-contig evidence and painting inputs, label expansion and O(K L) refit/gain arrays.
