# Pedigree direction and family evidence

The default T10 policy is `continuous_family`: finite chromosome-paired
orientation evidence plus four synchronous reciprocal-family cavity-message
passes. Tier B remains the primary product. This changes pedigree decisions,
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
back to itself as new support. Four passes are the default; this is bounded
loopy belief propagation, not exact global pedigree marginalization and not a
claim of convergence. Included/excluded configuration masses are retained as
log odds until the reverse message is removed, so very small probabilities
are not rounded away before the cavity calculation. The existing acyclic graph selection and graph-conflict
release checks remain in place.

All direction quantities and messages are recomputed in every chromosome
bootstrap and leave-one-chromosome-out fit. There is no fixed full-data family
scaffold, inferred root-count target, generation label, age, sex or sample-name
rule in metadata-free simulation inference. Explicit caller eligibility remains
mandatory, including exclusions of outside-pedigree real samples.

With `A` scored configurations, `C` chromosomes and `N` samples, the additional
work per resample is `O(C*N^2 + T*(A + N^2))`, with `T=4`. Extra message storage
is `O(A + N^2)`. With the fixed M2 panel, `A=O(N^2)`. No exponential pedigree
enumeration or higher-order founder-state HMM is introduced.

## Configuration and interpretation

The normal workflow builder selects:

```python
PedigreeConfig(
    parent_state_direction_model="continuous_family",
    parent_state_family_message_passes=4,
)
```

For controlled Python-API comparisons, `parent_state_direction_model="cluster"`
selects the previous policy. `continuous` and `family` are component ablations,
not the production recommendation. These are configuration/API settings, not
new CLI flags. Existing hard exposure requirements and Tier-A/Tier-B thresholds
are unchanged.

Diagnostics record `DirectionModel`, `FamilyMessagePasses`, and the resampling
method. The continuous model reports no fictitious discrete ancestry layer.
Source/configuration identities distinguish the new T10 checkpoints; accepted
historical results are not overwritten or relabelled by the validation runs.
Stage 11 continues to consume the same Tier-B relationship table and binds its
checkpoint identity to that table.

The genetic and painting-derived evidence share data. Neither message weights,
state support nor bootstrap fractions are calibrated probabilities of correct
parentage. Long feedback loops, incomplete candidate panels, unsampled parents,
closely related alternatives and highly structured missingness can still cause
ambiguity. See the [validation record](validation.md#finite-family-direction)
for the actual tested designs and their limits.
