# Pedigree direction and family evidence

The default pedigree policy is `continuous_family`: finite chromosome-paired
orientation evidence plus four synchronous reciprocal-family cavity-message
passes, a joint-family check of reverse two-edge ancestry paths, and a final
converged reciprocal solve with that path evidence frozen. Full-marker
Mendelian exclusion can then resolve M0/M1 counts conditional on already-supported
family directions, with separate diagnostics from chromosome resampling.
Tier B remains the primary product. This changes pedigree decisions,
not block discovery, L1–L4 assembly, painting, genetic likelihood scoring,
candidate eligibility, or the top-20 parent-pair panel.

## Why direction is not a hard ancestry-layer gate

Ancestry junction burden is not sequencing depth or known generation. Related
individuals can occupy the same fitted ancestry layer; a backcross offspring
need not have more junctions than both parents. The previous layer-order gate
could therefore remove a strongly supported true pair before identity selection.

The continuous term compares child-minus-parent junction burden across paired
physical chromosomes. Callability adjusts the counts; an inclusion-exclusion
bound weights potential common exposure. This is a summary-level approximation,
not an exact recount on intersecting genomic intervals. It assumes missingness
is sufficiently representative for that adjustment; structured missingness is
still a limitation. Absent overlap supplies neutral evidence.

A robust, neutral-centred log correction is added for each proposed parent:

`log(2 * ((1 - epsilon) * orientation_support + epsilon / 2))`.

`epsilon` uses the existing parent-state contamination setting (default 0.02).
Neutral support contributes zero, not an implicit penalty for observing a
parent. Even strong reverse support has finite weight; it is not an ancestry
order veto. Caller-supplied chronology overrides this extra orientation term.
The support is a composite score, not a calibrated parenthood probability.

## Family comparisons without immediate feedback

For each fish, retain competing M0, M1 and M2 configurations. M0 means no parents
among the observed candidates; an M0 fish can still have sampled offspring.
Configuration factors use the genetic score plus finite orientation evidence,
normalized by the full eligible identity count for that parent state. Family
messages use equal M0/M1/M2 weights; the existing final state priors and their
sensitivity checks remain separate and unchanged.

For a proposed parent `p -> c`, the incoming family message compares:

- configurations of `p` that **do not** also make `c` its parent;
- all configurations of `p`.

This marginalization includes possible co-parents rather than freezing a
previously selected pedigree as truth. Subsequent synchronous passes incorporate
other family comparisons, but remove the incoming reverse message before
forming the next message. Thus a fish cannot immediately send its own evidence
back to itself as new support. Four initial passes prepare the path factor;
the final solve stops at maximum log-message change `1e-8`. Try 128 undamped
iterations; on nonconvergence, restart the same equations with damping 0.5 for up
to 4096 iterations. The residual is measured before damping, so smaller updates
cannot falsely signal convergence. Successful initial solves are untouched.
Failure of both attempts is reported as an error, not silently published.
Convergence of these messages is not exact global pedigree marginalization. Included/excluded configuration masses are retained as
log odds until the reverse message is removed, so very small probabilities
are not rounded away before the cavity calculation. The existing acyclic graph selection and graph-conflict
release checks remain in place.

All direction quantities and messages are recomputed in every chromosome
bootstrap and leave-one-chromosome-out fit. There is no fixed full-data family
scaffold, inferred root-count target, generation label, age, sex or sample-name
rule in metadata-free simulation inference. Explicit caller eligibility remains
mandatory, including exclusions of outside-pedigree real samples.

## Joint short-ancestry paths

Direct reciprocity does not identify a proposed parent who is actually a
grandchild. This can matter after related individuals mate: a double grandchild
can share enough genetic material with a grandparent to resemble a direct
parent in chromosome resamples.

For a proposed child `c` with parental set `S`, the additional factor averages
over uncertain joint parent configurations of the members of `S`. Condition
each on not choosing `c` directly, since direct reciprocity is handled already.
For each distinct intermediate ancestor `g` in these configurations, multiply
the outgoing cavity probability that `g` does **not** choose `c` as parent.
Exclude members of `S` from these intermediates: a reverse path through the
other focal parent is already forbidden by reciprocity. Count shared
intermediates once. These distinctions preserve legitimate backcrosses.

Add the log averaged compatibility to both state and identity scores. It is
finite evidence, not a veto based on a called pedigree. The new term is not fed
back into the beliefs supporting it. After computing it once, replace the earlier
reciprocal term with a converged reciprocal solve on the path-adjusted
scores. This reconciles relationships whose competing configurations changed
after the path correction. Do not add both reciprocal terms, and do not
recompute the path factor from the final messages. Independent cavity-belief
approximations remain; removing immediate feedback does not remove every loopy dependence.

The calculation retains at most `R=16` parent configurations per fish, with
all omitted mass treated as compatible. This truncation weakens, never
strengthens, the full-enumeration penalty under the same approximate beliefs.
Ties at the panel boundary join the neutral tail, avoiding sample-order bias.

Explicit caller chronology exempts that focal parental side from the added
path penalty, just as it overrides the existing direction/reciprocal terms.
An uncertain co-parent is still evaluated, conditioning on both focal parents.
Chronology does not assert that a specific candidate really is a parent.
Eligibility masks remain authoritative and unchanged.

With `A` scored configurations, `A2` two-parent rows, `C` chromosomes and
`N` samples, the extra direction/family work per resample is
`O(C*N^2 + (4+T)*(A + N^2) + R*A + R*N^2 + R^2*A2)`, with `T` final
iterations (at most 128 plus 4096 on numerical retry) and `R=16`.
Storage is `O(A + N^2 + N*R)`. The fixed M2 panel gives `A=O(N^2)` and
`A2=O(N)`, so this remains quadratic in sample count. No exponential pedigree
enumeration or higher-order founder-state HMM is introduced.

## Fixed-observed-genome M0/M1 release

The standard workflow additionally checks all eligible dyads of unresolved or
non-exact Tier-B individuals using full-marker raw genotype likelihoods. At each
jointly observed site, let `A` be the likelihood mixture over opposite homozygotes,
`C` the largest likelihood among Mendelian-compatible dyad genotypes, and `U` the
unrestricted maximum. With genotype replacement allowance `delta=0.01`,
`D=(1-delta)*C+delta*U` bounds the compatible read model. Products of
`1-b+b*A/D` use six fixed bets, averaged over fixed 1/5/20-Mb half-overlapping
windows and whole chromosomes. Missing sites contribute one. Genome evidence is
an average across chromosomes, never an unpenalized maximum over selected windows.
A dyad is excluded at `log(E) >= log(2*N/alpha)`, with `alpha=0.01`.

This is an exclusion bound conditional on calibrated likelihoods and independent
read errors across markers, not independent inheritance of SNPs. Mapping errors,
ascertainment and correlated errors can violate those assumptions. The `2*N`
allowance covers the maximum number of true observed parent edges, including when
unresolved targets were selected using the same data.

Mendelian compatibility is symmetric: it does not distinguish parent from child.
An already-supported outgoing edge rules out its reverse by acyclicity. Release
M0/M1 only when every eligible candidate has been checked, the remaining set
matches both the supported incoming edges and the full-data configuration, and
ordinary information/graph-conflict checks pass. No new edge is introduced. This
combined release additionally depends on those supported directions being correct;
**it is not an unconditional 99% pedigree-accuracy guarantee**. M0 still means no
observed eligible parent, not founder generation.

`tier_b_relationships`, its partial and candidate-set views, and call explanations
record `mendelian_exclusion_and_supported_direction` as the release evidence.
`exclusion_evidence.csv` and `exclusion_diagnostics.csv` retain dyad and decision
evidence. Original bootstrap/LOCO fractions and `TierBStateCall` are unchanged;
`TierBFinalStateCall` records the released final configuration. The fixed-genome
evidence is not added to HMM scores or frozen into bootstrap fits.

Full-marker checks cost `O(U*N*L)` for `U` unresolved individuals and `L` total
markers, worst-case quadratic in sample count. Up to four chromosome workers
share the allocated CPU budget, rebalancing between candidate batches, with
independent chromosome checkpoints. Existing assembly, painting, preparation and
genetic-score caches remain reusable; changed decisions get a new identity.
The score-only API has no raw markers and therefore does not perform this step;
normal checkpointed workflows do. Set `full_marker_exclusion=False` in the Python
configuration for a controlled ablation.

## Configuration and interpretation

The normal workflow builder selects:

```python
PedigreeConfig(
    parent_state_direction_model="continuous_family",
    parent_state_family_message_passes=4,
    parent_state_ancestry_path_budget=16,
    parent_state_family_final_max_iterations=128,
    parent_state_family_final_tolerance=1e-8,
    parent_state_family_retry_iterations=4096,
    parent_state_family_retry_damping=0.5,
    full_marker_exclusion=True,
    mendelian_exclusion_alpha=0.01,
    mendelian_genotype_replacement_probability=0.01,
)
```

For controlled Python-API comparisons, `parent_state_direction_model="cluster"`
selects the previous policy. `continuous` and `family` are component ablations,
not the production recommendation. Setting `parent_state_ancestry_path_budget=0`
is a path-only ablation; positive values bound its configuration panel. The
path correction and final reconciliation apply to the `family` and
`continuous_family` modes. A zero path budget omits only the path factor;
the final reciprocal messages still converge.
These are configuration/API settings, not
new CLI flags. Existing hard exposure requirements and Tier-A/Tier-B thresholds
are unchanged.

Diagnostics record `DirectionModel`, `FamilyMessagePasses` (path preparation),
`FamilyReconciliation`, `AncestryPathBudget`, and the resampling method. The continuous model reports no fictitious discrete ancestry layer.
Source/configuration identities distinguish the new pedigree checkpoints; accepted
historical results are not overwritten or relabelled by the validation runs.
The checkpoint records the final full-fit iteration count, retry use and residual
as `final_family_solve`. Family phase consumes supported Tier-B partial edges and
binds its checkpoint identity to those edges and their support flags.

The genetic and painting-derived evidence share data. Neither message weights,
state support nor bootstrap fractions are calibrated probabilities of correct
parentage. Long feedback loops, incomplete candidate panels, unsampled parents,
closely related alternatives and highly structured missingness can still cause
ambiguity. See the [validation record](validation.md#finite-family-direction)
for the actual tested designs and their limits. The subsequent
[joint-path validation](validation.md#joint-short-ancestry-paths) includes
deep-generation, backcross and sample-withholding controls.
