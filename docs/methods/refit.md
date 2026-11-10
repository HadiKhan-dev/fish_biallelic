# Founder allele refit

Refit uses released painting labels to reconsider individual founder alleles with untempered site evidence. Labels are held fixed: this is not another ancestry search. N is scoring samples, L sites and K founder rows. Tied incompatible updates release unknown alleles. Pedigree consumes the frozen pre-refit painting and does not receive this output as feedback.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="fr01"></a>
## fr01 Refit scoring set and founder-pair table (released labels expanded to sites, founder view, untempered robust logs)

Expand released painting labels onto sites and form untempered robust genotype evidence from the founder view. Score only samples admitted by the refit rule. Sites with no scoring sample retain their alleles; painting’s tempered bin objective is not reused as the site objective.

Implementation: [science/refit/refit.py](../../haplotype_reconstruction/science/refit/refit.py), [science/painting/release.py](../../haplotype_reconstruction/science/painting/release.py).

Parameters by name: `robustness_epsilon`, `REFIT_TIE_QUANTUM`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fr02"></a>
## fr02 Refit coordinate ascent: flip and two-row phases, rounds, interaction, sweeps, hysteresis gains

Within each site, run the named sweep schedule with flip rounds before two-row rounds. Rank qualified moves, withhold conflicting tied alternatives, otherwise apply independent moves and rescore affected candidates. Limit each row’s changes as declared. For finite tables, score a two-row move by the identity `joint_gain = flip_gain(a) + flip_gain(b) + Delta_ab`, where `Delta_ab` is the pair curvature. Cached single-row gains make each pair score O(1). A table containing an infinite entry (a zero likelihood at robustness epsilon 0) uses direct gains and conflict checks, because the identity can evaluate infinity minus infinity even for a finite swap gain. The minimum gain is per-move hysteresis per changed cell, not a fixed prior on the assembled allele.


Implementation: [science/refit/refit.py](../../haplotype_reconstruction/science/refit/refit.py).

Parameters by name: `minimum_gain_nats`, `coordinate_sweeps`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K^2); O(P); O(n). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fr03"></a>
## fr03 Refit tie classes and decision order (carrier-scaled quanta, wide classes, convention)

Sort exact gains stably, form carrier-scaled tie components and identify wide classes. Stable comparison sorting is bounded by `STABLE_SORT_COMPARISON_LIMIT` keys (depth at most 12); larger arrays use eight fixed-width radix passes over IEEE bit images. Depth-first labelling of at most 2(n − 1) interval-relation edges forms the same tie partition. Counting placement gives content order inside classes. Ordinary classes use the row/pair convention; wide classes retain value order as distinct choices. Reversal changes order within allowed classes, never their membership.

Implementation: [science/refit/refit.py](../../haplotype_reconstruction/science/refit/refit.py).

Parameters by name: `REFIT_TIE_QUANTUM`, `REFIT_TIE_QUANTUM_PER_CARRIER`, `WIDE_CLASS_QUANTA`, `STABLE_SORT_COMPARISON_LIMIT`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(n). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fr04"></a>
## fr04 Refit ambiguity release (tied mutually exclusive moves released unknown)

For mutually interacting moves in one tie class, apply each hypothetically and test whether the other still qualifies. Mark every conflicting pair in one pass and release all involved alleles as unknown before rescoring. A called allele withheld by this ambiguity cannot be filled later in the site.

For finite tables, the conflict test uses O(1) gain updates: after a move flipping rows V, another move's gain changes by the sum of its pair curvatures with V. Two candidates sharing a row exclude each other because applying either invalidates the other's move kind; mark their endpoints directly.

Let T contain the endpoints of members of every tie class with at least two members. All these rows are called and unswapped in the current two-row round. For each v in T, take `m_T(v) = min(0, min_{u in T, u != v} Delta_vu)`. Candidate i with endpoints a,b is certified when `gain_i + 2(m_T(a) + m_T(b))` passes both the positive-gain and minimum-gain gates. For disjoint members i,j, their cross-curvature sum W is at least each member's bound. Two certified members therefore cannot conflict; every other relevant pair remains examined. Shared-row marks always apply. Every class is examined before any move is applied, and all marked rows are released together. An independent serial reference in the external test suite retains direct gains and exhaustive conflict tests.

The identities and certificate preserve the mathematical rule. Floating-point threshold or tie decisions can differ from direct sums at rounding-level equalities. A conservative gain allowance is `2 (2K + 20)^2 eps_machine max(1, max|table|)` nats (twice this for comparisons of two gains). Witnesses pin the accepted production decisions and record the direct-reference margin where they differ.

With S coordinate sweeps, finite-table work per site is `O(N + (S + 1) K^3 + K sum_r Uu_r)`. Here Uu_r counts uncertified tied candidates sharing no row with another member of their class; the sum includes every applying and releasing two-row round. Base scoring, ordering and certification cost O(K²) per round. Scans between unshared members cost at most K Uu_r; scans against shared members total O(K³) across the site because each releasing round removes their rows permanently. If all tied members are certified, the sum is zero. For fixed S, supercubic work requires that sum to exceed order K²; order K uncertified members per round over order K rounds still fits cubic work. The retained worst-case exception is O(K³) per round and O(K⁴) per sweep. Non-finite tables use the direct route, up to O(K⁵) in one round.

Implementation: [science/refit/refit.py](../../haplotype_reconstruction/science/refit/refit.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: finite tables O(N + (S + 1) K^3 + K sum_r Uu_r) per site; at most O(K^3) per round and O(K^4) per sweep. Non-finite tables: up to O(K^5) per round. These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fr05"></a>
## fr05 Refit fill rule for alleles the painted panel left unknown

After the coordinate sweeps, visit alleles that were already unknown in the painted panel. Test the declared allele order, minimum gain and supporter gates inclusively. Preserve unknown on insufficient evidence; this fill cannot undo ambiguity withholding of an originally called allele.

Implementation: [science/refit/refit.py](../../haplotype_reconstruction/science/refit/refit.py).

Parameters by name: `minimum_gain_nats`, `fill_min_supporters`, `REFIT_TIE_QUANTUM`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(K^2). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="fr06"></a>
## fr06 Refit stage, product, records and downstream reach

Publish refit component panels and diagnostics under evidence/painting/parameter identities. Resume only compatible records. Evaluation reads the original refit axes; export consumes the separate release/withholding view. Pedigree continues to use its pre-refit founders.

Implementation: [stages/refit/tasks.py](../../haplotype_reconstruction/stages/refit/tasks.py), [science/refit/run.py](../../haplotype_reconstruction/science/refit/run.py), [stages/refit/records.py](../../haplotype_reconstruction/stages/refit/records.py), [contracts/refit.py](../../haplotype_reconstruction/contracts/refit.py).

Parameters by name: `SITE_TILE`. Values, units and current rationale are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: Per-contig evidence and painting inputs, label expansion and O(K L) refit/gain arrays.
