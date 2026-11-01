# Shared records and numerical rules

Typed records distinguish observations, evidence, inference state, released calls, reports and diagnostics. Shared numerical and tie functions define one expression for each named quantity. They do not choose a stage’s model, candidate set or search order. Array axes, dtypes, units and view lifetimes are explicit boundary obligations.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="ec28"></a>
## ec28 Local panel type and the unknown-copy mass

Represent each local panel with separate latent/released alleles, posterior probabilities, support and unknown-copy information. Unknown-copy mass is a derived view of the declared uncertainty, not another allele fit. Validate axes/dtypes at the typed boundary.

Implementation: [contracts/panels.py](../../haplotype_reconstruction/contracts/panels.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(K L) panel validation/canonical representation per block.

<a id="ec29"></a>
## ec29 Founder component types: prepared-panel digest, materialisation, relabelling, shared latent alleles

Materialise component row paths over immutable prepared panels and carry row relabellings consistently. Prepared-panel digests bind the content used by join reuse. Shared latent alleles and released calls are distinct views; a cache cannot replace one with the other.

Implementation: [contracts/panels.py](../../haplotype_reconstruction/contracts/panels.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(K L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec30"></a>
## ec30 Tie quanta: kinds, noise motions, noise_quantum, BlockQuanta, block_scale

Construct named tie quanta from their noise motion, scale and floor, rounding as each policy declares. Compute a block scale once from its evidence. The policy catalogue distinguishes NLL, allele-step, improvement, sample, cell and probability comparisons; one generic epsilon cannot replace them.

Implementation: [core/ties.py](../../haplotype_reconstruction/core/ties.py).

Parameters by name: `WIDE_CLASS_QUANTA`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(1); O(N L_block). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec31"></a>
## ec31 Anchored tie classes, decision order, guards and TieCounter

Sort values stably, join consecutive values into anchored tie classes according to the policy, and record boundary/width events. Apply secondary/content ordering inside ordinary classes and value-first ordering in wide classes. Equal non-finite endpoints and NaNs follow explicit guards; reversal never changes class membership.

Implementation: [core/ties.py](../../haplotype_reconstruction/core/ties.py).

Parameters by name: `WIDE_CLASS_QUANTA`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n log n). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec32"></a>
## ec32 Exact decision helpers: improves, threshold_band, inverse_permutation

Apply the shared strict improvement test against the incumbent plus quantum, the declared threshold-band test, and exact inverse-permutation construction. Equality is not improvement. These helpers express different decisions and cannot be substituted for each other.

Implementation: [core/ties.py](../../haplotype_reconstruction/core/ties.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(1). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec33"></a>
## ec33 Convention reversal  and the conv_* helpers

Pass TieSettings into declared convention policies (L8). Reverse only arbitrary content/index/incumbent conventions that the policy permits, retaining model priors and fixed preferences. Store the setting in the relevant derivation; no scientific function reads environment switches.

Implementation: [core/ties.py](../../haplotype_reconstruction/core/ties.py), [store/identity.py](../../haplotype_reconstruction/store/identity.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(1). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ec34"></a>
## ec34 Valley census

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Deliver comparison values to an explicitly supplied census sink and return typed tie counters. Disabled collection avoids diagnostic work without changing the decision. Census output is outside scientific identity and must not be read back as a threshold calibration during a run.

Implementation: [core/ties.py](../../haplotype_reconstruction/core/ties.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Constant lookup when recording is disabled; otherwise proportional to emitted events.

<a id="ec35"></a>
## ec35 Founder-pair genotype emissions (pair_code, predictive likelihood, robust centred log)

Map founder allele pairs to genotype codes and use the shared predictive/robust-centred emission formulas. Keep missing evidence neutral and the declared operand order. Consumers share one implementation so a copied arithmetic variant cannot silently change a decision.

Implementation: [science/shared/emissions.py](../../haplotype_reconstruction/science/shared/emissions.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(1). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.
