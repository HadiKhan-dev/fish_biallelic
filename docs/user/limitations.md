# Limitations and interpretation

The documented scale envelope targets up to **N = 2,500 samples on bounded crops**
and **K = 30 founder haplotypes**. Crop-based evidence does not establish that every
whole-genome dataset, chromosome length or missingness pattern will fit a given
machine. Read the [cost table](../developer/costs.md) for the actual validation
scope, memory terms and measured results.

## Biological meaning

Sample names identify rows. No generation, cohort, sex or sample-metadata file
enters inference. A sequenced founder-cohort individual is not automatically a
known parent, and a cohort label is not individual-level truth. Parent slots are
unordered conventions and carry no sex. Some parents may be unsequenced, ambiguous
or unresolved; inspect parent-state calls and candidate sets alongside exact edges.

Founder rows are inferred haplotypes within a component. Their labels are not
immutable chromosome-wide biological identities. Breaks, continuation ambiguity
and identity flags qualify comparisons across components. Missing alleles and
unobserved-carrier withholding are preserved rather than imputed.

A supplied genetic map is a model input, not observed truth. Recombination output
is conditional on the fixed inferred pedigree and final phase. Uninformative gaps
are not evidence for an absence of recombination. Neither family phase nor the
inferred map feeds back into founder or pedigree inference.

## Model and search approximations

Discovery and assembly fit finite candidate panels with the specified search
orders and budgets. Pedigree uses a bounded M2 candidate panel, composite evidence,
chromosome-level calibration and release rules; support fractions are not a claim
that all possible joint pedigrees were enumerated. Family phase releases modular,
conditional calls rather than a single globally identified biological haplotype
assignment. Missingness and ambiguity can remain after a converged fit.

The [methods pages](../methods/index.md) name each carried rule and approximation.
The following cost exceptions are explicit in 4.0:

| Exception | Retained behaviour | Meaning at K = 30 |
|---|---|---|
| 0 | Rare context-coded local-selection screening, O(K⁵) per round | K⁵ is 24,300,000; this is an algebraic factor, not a measured operation count or runtime |
| 0b | The assembly joins' fixed `ranked_refits` candidate cap | The cap does not grow to enumerate every candidate at larger K; it is a documented search limitation |
| 0c | Discovery fitting O(N L K⁴), bounded replacement/archive/iteration searches; local-selection backstop O(N L K³ (K + L)) and context snaps O((K + L)²) passes | K⁴ is 810,000 and K³ is 27,000; full cost also depends on sample/site counts and the recorded search path |
| 0d | Optional pairwise IBD output, O(N² L) | Independent of the founder-count ceiling; at N = 2,500 there are 3,123,750 unordered sample pairs |
| 0e | Four worst-case costs above cubic in K: assembly joins' structural screening, refinement window fixing, painting label-continuity checks, founder-refit ties | Carried exactly and listed for review in 4.1; none binds on the validation inputs; see the [cost page](../developer/costs.md#exemptions-and-exceptions) |

<!-- cost verdicts: generated from the cost table; do not edit -->

The [cost table](../developer/costs.md) gives a K gate for each of its 37 stage rows: 21 at most cubic in their probed kernels, 11 under a decided exemption or exception, 5 with no intrinsic K, 0 pending; 2 of these are provisional until their package lands. The exempted rows are:

| Stage | Exemption | At K = 30 |
|---|---|---|
| [relatedness.pairs](../methods/relatedness.md) | exemption 0d | N2500:3123750 pairs,224910000 bytes; off by default |
| [discovery.fit](../methods/discovery.md) | exemption 0c | K⁴=810000; 50 iterations; ≤2 trajectories/start |
| [discovery.search](../methods/discovery.md) | exemption 0c | 810000 N L; exact switch 12; 48/96 replacements; 6 panels/K; 50 iterations |
| [selection.fit](../methods/selection.md) | exemption 0c | K²=900; snaps(140+2L)²; L400 release 393625600 bytes |
| [selection.search](../methods/selection.md) | exemptions 0 and 0c | K⁵=24300000/round; 27000 N L(30+L); N2500 B100 S31=1922000000 bytes |
| [assembly.joins](../methods/assembly.md) | above cubic: exception 0e ([AJ17](../methods/assembly.md#aj17)) | K30 N2500 m2000 emissions≈18.6GB; direct-route finite-grid quartic trend carried; 4.1 candidate |
| [assembly.shadow_refits](../methods/assembly.md) | exemption 0b; bounded shadow family | 9 refits; 2048×900=1843200 N m; ≈1.3GB/refit N2500 |
| [refinement.search](../methods/refinement.md) | above cubic: exception 0e ([RS05](../methods/refinement.md#rs05), [RS07](../methods/refinement.md#rs07)) | K30 N2500 polish≈3.7GB; canonical-fixing worst case carried; 4.1 candidate |
| [refinement.count](../methods/refinement.md) | exemption 0b; bounded shadow family | K30 N2500 B641 copy≈19GB; ≤2048 shadow scores/member |
| [painting.coalescence](../methods/painting.md) | above cubic: exception 0e ([PR17](../methods/painting.md#pr17)) | N2500 K30 links 2000≈1e11 gain operations; quartic candidate checks carried; 4.1 candidate |
| [founder_refit](../methods/refit.md) | above cubic: exception 0e ([FR04](../methods/refit.md#fr04)) | K⁴=810000; K⁵=24300000/site/phase; observed≤3 rounds,≤2 tie members; 4.1 candidate |

Kernel slopes cover the probed kernels only; an exemption's worst case is analytic.

<!-- end cost verdicts -->

The parameter reference gives the exact caps, defaults and provenance. The
[accretion register](../developer/accretions.md) distinguishes retained diagnostics,
accepted exceptions and candidates for a later scientific revision. They are not
silently replaced to meet a performance target.

## Validate the intended use

Use simulations with independent recorded truth to measure accuracy. Without
truth, inspect competing-candidate margins, chromosome consistency, missing-parent
states, unresolved calls and legitimate negative controls. A better headline
metric alone does not validate a model. The quick-start example checks wiring;
it is too small and artificial to establish performance or pedigree accuracy.
