# Pedigree direction and family evidence

The default pedigree policy is `continuous_family`: per-chromosome junction-count
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

The continuous term is a count likelihood of the painted minimum junction
counts `J` (founder-label switches of both homologues), evaluated chromosome by
chromosome. A homologue that parent `p` transmits copies one of `p`'s two
homologues at each position, so it carries each of `p`'s painted junctions with
probability 1/2; the child shows it where the child is callable (fraction `f_c`
of its haplotype bins, missingness assumed representative). Everything else in
the child's count — the new crossovers of the two meioses that formed it, the
junctions transmitted by the other parent, and painting false positives — is a
non-negative remainder with no assumed distribution. On chromosome `k`:

`L_k(p -> c)` is proportional to `P(Binomial(J_p, f_c / 2) <= J_c)`, and
`L_k(c -> p)` to `P(Binomial(J_c, f_p / 2) <= J_p)`.

Under the reverse orientation the proposed parent would have to have lost
junctions that its own "parent" transmits. Equal counts give no evidence; a
junction-poor parent of a junction-rich child gives strong evidence. With
probability `epsilon_c` (`parent_state_direction_chromosome_error_probability`,
default 0.05) one chromosome's comparison is painting failure carrying no
orientation evidence:

`pi_k = (1 - epsilon_c) * L_k(p -> c) / (L_k(p -> c) + L_k(c -> p)) + epsilon_c / 2`,

so each chromosome contributes at most `log(2 / epsilon_c - 1)` (3.7 nats).
The default is the upper range of gross painted-versus-true junction
discrepancies (at least 3 junctions on one chromosome in either sample: 0.6-8%
of pair-chromosomes in the std, stress and real-axis simulations of October
2026). Chromosomes are independent meioses, so their log odds add:
`LLR = sum_k logit(pi_k)`, weighted by chromosome multiplicity in bootstrap and
LOCO fits. Painting failure is not always independent between chromosomes: a
sample-level junction excess or deficit repeats on every chromosome, which the
per-chromosome floors cannot bound. In the real 22-chromosome Tropheops run
(metadata-free replay) one founder-generation sample carried 362 painted
junctions and another 39, and painted parents carried as many junctions as
their children on average, so unbounded sums reached -25 to -32 nats against
the recorded generations. With probability `epsilon` (the parent-state
contamination setting, default 0.02) the pair's whole comparison is therefore
painting failure carrying no orientation evidence, and the neutral-centred log
correction added for each proposed parent is

`log(2 * ((1 - epsilon) * sigmoid(LLR) + epsilon / 2))`.

Neutral evidence contributes zero, not an implicit penalty for observing a
parent; supported orientation gains at most `log(2 - epsilon)` and a
contradicted one costs at most `log(epsilon)` (3.9 nats), the bound of the
earlier contrast term. The count likelihood changes how quickly evidence
accumulates within that bound, not the bound. It is not an ancestry order veto.
Caller-supplied chronology overrides this term.
The expected number of new crossovers is deliberately not imposed: it depends on
the recombination rate (the inference prior can differ from the generating
rate, 5 versus 2 cM/Mb in the real-axis simulations), on whether the
transmitting parent's homologues differ at the crossover, and on painting
sensitivity; imposing it oriented low-count true edges backwards in those
simulations. Related individuals of equal junction burden (full sibs, a
backcross offspring of a deep parent) therefore receive little orientation
evidence, as they should. The support is a composite score, not a calibrated
parenthood probability. The earlier paired junction-density contrast is kept
only to order the acyclic graph search and as the reported parent-role
probability.

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
the final solve stops at maximum log-message change `1e-8`. Try 8192 undamped
iterations (slow monotone modes need thousands; damping would halve their rate);
on nonconvergence, restart the same equations with damping 0.5 for up
to 4096 iterations. The residual is measured before damping, so smaller updates
cannot falsely signal convergence. Successful initial solves are untouched.
Failure of both attempts in the full-data fit is reported as an error, not
silently published. A bootstrap or LOCO resample whose solve fails abstains for
every fish, which only lowers support, and is counted in `final_family_solve`.
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
`O(C*N^2*m + (4+T)*(A + N^2) + R*A + R*N^2 + R^2*A2)`, with `m` the typical
per-chromosome junction count (binomial tail sums), `T` final
iterations (at most 8192 plus 4096 on numerical retry) and `R=16`.
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

## Orientation of released edges

Chromosome resampling measures stability, not orientation. Two strongly linked
fish whose junction counts cannot orient them (a parent's transmitted homologue
with one crossover genome-wide, a child whose painting lost junctions) can be
released in the wrong direction with support just above the Tier-B cut-offs,
because each resample picks one of two nearly equal reciprocal configurations.
A released inferred edge `p -> c` therefore also needs a decisive full-data
orientation. With every other converged reciprocal message held fixed, the
pair's anti-reciprocity factor is evaluated exactly: for the cavity inclusion
log odds `a` (c's configurations naming p) and `b` (p's configurations naming
c), the joint states `p -> c`, `c -> p` and neither weigh `e^a : e^b : 1`. The
edge is released only when `a - b >= log((1 - q) / q)`, with `q =
orientation_release_reversal_probability` (default 0.02, 3.9 nats, the
contamination floor). Otherwise Tier A/B keep a supported state with status
`..._parent_state_supported_orientation_unresolved`, and the partial and
candidate-set views omit the edge, so family phase never conditions on it. The
complete table keeps the full-data best guess. Caller-supplied chronology is
exempt, so metadata-supported workflows are unchanged. In the six-chromosome
real-axis simulation without coin-flip templates (October 2026), the reversed
edge `F1_6 <- F2_67` had `a - b = 1.2` nats, and every released true edge had at
least 5.9.

## Configuration and interpretation

The normal workflow builder selects:

```python
PedigreeConfig(
    parent_state_direction_model="continuous_family",
    parent_state_direction_chromosome_error_probability=0.05,
    orientation_release_reversal_probability=0.02,
    parent_state_family_message_passes=4,
    parent_state_ancestry_path_budget=16,
    parent_state_family_final_max_iterations=8192,
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
are unchanged; the orientation condition above is an additional release check.

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
