# Scientific assumptions and output interpretation

## Missingness and phase

Alleles use 0/1 for reference/alternate and -1 for unknown. Raw genotype
likelihoods and the observed-read mask are separate: a flat likelihood at a
missing site is not an observed heterozygote. Discovery pools compatible
sample evidence at each marker, so a representative can be called where some
carriers are missing and others are informative. Unsupported positions remain
unknown; missingness does not count as an allele match.

Assembly carries observation support and preserves disconnected components.
Painting uses the locally available founder states and an unknown/unrepresented
state. A sole discovered haplotype does not force every sample to be homozygous
for it. Phase is local to supported components, and H1/H2 names are arbitrary
orientation labels, not paternal/maternal identities for pedigree roots.

## Local feedback and candidate selection

Before final chromosome assembly, L1 context is projected/refitted back to the
original 200-SNP blocks and selected locally; the selected panels then supply
the L1+L2 context for a second refit/selection round. Original candidates remain
available in both rounds. Context excludes the target block's emission when
estimating carrier weights. Candidate panels are starts, not extra read evidence.

The default balanced selector combines a cavity-ranked feedback backbone with
BIC-supported candidate additions not explainable by one donor join, followed
by same-K cavity-ranked refinement. The strict option limits rescue to private
alleles and protects each round's backbone during that rescue. Both use the
original likelihoods and observation masks, and release unsupported alleles as
unknown. These selection scores are not calibrated correctness probabilities;
novelty filtering is a heuristic, not proof of a distinct biological founder.
After each feedback selection, an entirely uncalled row triggers a bounded
smaller-panel comparison. Single-row deletions (and deletion of all empty rows
when nonempty rows remain) are refitted against the original observed likelihoods
and accepted only if the existing BIC-like score does not worsen. Assignments,
site support, probabilities and released calls are rebuilt together. Balanced
selection permits the usual allele/assignment refit; strict selection additionally
protects surviving backbone calls. Partially called rows do not trigger deletion,
and an empty row whose deletion worsens the score remains explicit uncertainty. A final
single unknown row is not converted into an invalid zero-founder panel.

See [configuration and checkpointing](running.md#local-feedback-selection) and
[local validation](validation.md#local-feedback-selection).

## Hierarchical linking

All four assembly levels use the same distance-aware normal/error-tract linker
in `assembly/linking.py`, with at most 20 EM iterations per block gap and the
same convergence rule. The linker uses the input genetic map where supplied,
or the configured scalar rate otherwise. L1 keeps its existing block grouping
and unlimited beam-gap rule; higher levels retain their distance-based beam-gap
limit. Missing-data component boundaries and founder-source provenance are
preserved at every level. Large linking proxies are sampled from kept markers
called in at least one frozen founder candidate; sample eligibility and boundary
support are checked on those actual proxies. Wholly unsupported blocks remain
unresolved, and explicit component breaks are not bridged.

Partial sites use a fixed Bernoulli(1/2) predictive distribution for each unknown
founder allele. A known 0 plus an unknown averages genotype likelihoods 0 and 1;
a known 1 plus an unknown averages 1 and 2. Two different unknown founders use
weights (1/4, 1/2, 1/4). Two copies of the same unknown atomic founder use
(1/2, 0, 1/2): they share one allele, including when different assembled paths
traverse the same local source row. This is a per-observation predictive
approximation, not exact integration of shared latent alleles across samples.
It changes linkage evidence, never imputes a released founder allele. It can
prefer uncertain paths in ambiguous regions and is not a confidence guarantee.

The binned panel scorer uses the same partial-founder distributions, applying
its existing likelihood floor after marginalization. All candidate states use
the same retained markers. Uniform sample evidence and all-founder-unknown
markers are neutral. Fully called input retains the previous optimized kernels.
Final founder refinement retains its separate fixed complete-site acceptance
objective described below; partial proposal scoring does not relax that guard.

The implementation stores three genotype emissions for complete input or seven
predictive categories for partial input per sample/site, reuses
state-constant-prior scans, omits unused terminal messages, and fits only mesh
gaps consumed by the beam. Homologue edge counts use a cubic contraction with a
cubic log-domain fallback; no quartic diploid transition tensor is needed. These
optimizations preserve the current fitting rule and its stopping criterion.

The within-block scans use sum-product inference: both ancestry and quality
alternatives are marginalized, not selected by maximization. Normal and error
states share ancestry transitions. The quality state follows a continuous-distance
Markov process in physical base pairs, with a default stationary error fraction
of 0.01 and mean error-tract length of 20,000 bp (`LINKER_ERROR_FRACTION` and
`LINKER_ERROR_TRACT_BP` in `core/config.py`). These are model priors, not measured
error rates. Quality starts at stationarity independently in each block and is
summed out at its opposite boundary; it is not propagated across block boundaries.
Genotype emissions use a 1% uniform mixture:
`log(0.99 * genotype_likelihood + 0.01 / 3)`, with no additional log-likelihood
clipping. The mixture itself bounds contradictory evidence at approximately
-5.70 for normalized likelihoods. The quality/error-tract process remains
enabled; removing the extra floor does not remove robustness to correlated
errors. This is the sole **linker** emission model for all four assembly levels;
panel selection and the final founder refiner have their separate objective below.

The error emission averages the three robust genotype likelihood weights,
independently of founder identity. Flat/missing observations therefore remain
neutral, including under common per-site likelihood rescaling.

The ancestry model retains the existing at-most-one-homologue-switch
approximation between retained markers, with its relative weights normalized to
unit row mass. The forward and backward micro-scans are adjoints of the same
block operator. Scaled arithmetic and the log-domain numerical fallback implement
the same model, both quadratic in founder count per sample and marker.

The macro linker uses the same forward transition factors in both likelihood
directions and fits ordinary expected homologue counts once per edge. Initial
diploid priors are normalized and uniform for each gap/residue chain. The
backward contraction uses the transpose of the forward transition matrix,
without renormalizing it into a reverse conditional. This supports unequal
haplotype counts in neighbouring blocks.

Pseudocounts, the probability floor, uniform robustness and forward parameter
damping regularize the fitted transitions. Reverse summaries for mesh/path
selection are column-normalizations of the same regularized edge evidence, not
likelihood operators or unconditional time-reversed chain transitions. This is
the default dense linker at every hierarchy level. Both assembly modes default
to bounded candidate-panel search, with full Viterbi/BIC scoring before accepting
a proposal; `--assembly-search broad` retains the optimized broader search.
The optional [structured model](founder_scaling.md) restricts the
macro transition to sparse specific edges plus a positive shared background.
It does not change discovery or the within-block emission model. Dense
propagation is cubic in founder count; structured propagation is near quadratic
with bounded search and fixed fitting budgets. Broad search adds further
rescoring costs. Numerically guarded matrix multiplication
accelerates larger dense contractions without restricting their parameters.

Coherence here refers to each independently fitted gap/residue chain. Combining
overlapping mesh edges in the beam remains a separate heuristic, and final
beam/path selection uses maximization. Neither the chain model nor its
robustness adjustments imply globally calibrated assembly path probabilities
or guaranteed monotonic unpenalized likelihood. Correlated corruption that
conceals a genuine crossover remains an identifiability and confidence-calibration
limitation.

Assembly checkpoint identities record the coherent expected-count linker,
partial-founder predictive emissions, empty-row refitting, unclipped mixture
emissions and iteration cap. Incompatible assemblies are not
silently reused; changing the linker requires new assembly/downstream identities,
not regeneration of the underlying simulated reads or discovered blocks.

## Final founder-path refinement

The hierarchy chooses component boundaries and an initial founder count. Final refinement
then reopens row choices from the original **prepared local panels** within each
component. A locally supported founder can otherwise be pruned at L2 and remain
unrecoverable to L3/L4, even when a better chromosome path exists in those local
panels. Refinement changes whole local-row selections, not their allele values,
support metadata, or pre-fill inference snapshots. It neither crosses component
breaks nor adds founders, and runs only for final assembly (`max_level=4`), not
the L1/L2 feedback contexts.

This pass uses the full cohort's genotype likelihoods, without truth, pedigree
or generation labels. It retains the panel scorer's normalized likelihoods,
1% uniform mixture, -2 log-likelihood floor, and length-scaled uniform penalty
for a change of sample diplotype. Unlike the binned panel scorer, acceptance
permits sample state changes at every SNP. This is a finer-discretization model
change, not merely a faster evaluation of the binned model. It is an internal
assembly fitting HMM, not the homologue-specific T09 painter, a posterior phase
confidence, or a recombination-map estimator.

Fixed-painting row edits provide cheap proposals. Conditional beams vary one
founder while retaining every sample's diploid state against the other founders;
an incumbent suffix supplies complete-path ranking in both scan directions.
The first beam sees the original assembly and competes with warm proposals,
rather than inheriting a potentially worse warm-start search basin. Every
accepted proposal improves the same full-site, fixed-count objective.

If fine-scale proposals stall, paths from the **final hierarchy's L1 level**
supply larger competing moves alongside the incumbent path pieces. These are
not a replacement of the local panels or another feedback round. Exact duplicate
row sequences are removed; a bounded beam joins these pieces, then decodes the
proposal back to original prepared rows. All original emission bins, sample
state transitions and evidence masks remain unchanged. Accepted larger moves
are followed by the ordinary fine-scale refinement.

Larger moves have an additional genotype-fit guard. Write the optimized sample
score as `Q = E - penalty * S`, where E is the cohort's centered genotype-emission
score and S counts internal sample diplotype changes. A macro move must improve
Q and must not lower E beyond numerical tolerance. This prevents accepting a
long founder rewrite solely because it saves switch penalties while fitting
the genotypes worse. It is a search-policy restriction, not a confidence
threshold or proof of biological correctness; a true move that sacrifices
some genotype fit can be withheld. Existing fine-scale proposals are unchanged.

The first pass starts with width 64, widens fourfold up to 1024 when the best beam
improves on the cheaper proposal by more than one switch penalty, and retains
that budget for subsequent sweeps. This is a computational heuristic, not a
confidence threshold. There are at most 20 sweeps, one focal path per sweep and
16 candidate local rows per beam expansion. Unknown observations are neutral;
the evidence mask is fixed across all original local candidate rows, so changing
a path cannot improve its score by hiding difficult sites. Original missing
calls can still move with the selected row.

After that pass completes, a second, default-on escape pass uses chain dual
decomposition. Each sample temporarily has its own copy of the focal founder's
local choices. Zero-sum messages equalize conditional max-marginals across
samples; alternating forward/reverse sweeps decode a **single common founder
path**. Relaxed sample-specific founder copies are never released. The incumbent
is retained unless a feasible decoded path improves its binned score, and the
same full-site score and larger-move genotype-fit guard determine acceptance.
This changes proposal search, not the likelihood or calling thresholds.

The escape pass considers all current founders, both scan directions, original
local rows and the same L1 pieces; it starts from the completed first pass, not
raw L4. There are at most 20 outer refinement iterations and 20 dual sweeps per
proposal, with the same 16-row branch cap. Width-independent dual proposals are
not retried at larger nominal beam widths. The remaining positive dual gap is
an optimization diagnostic for the restricted binned conditional problem, not
a biological confidence interval or proof of a globally optimal assembly.
The dual pass itself cannot recover a missing candidate allele or change count.

The following default-on searches address different remaining barriers:

1. Paired suffix exchanges change two founder paths together. Forward/backward
   messages score these relabelings without a full chromosome refit per
   boundary. In the pre-count phase pass, the best bounded proposals receive
   canonical full-site rescoring and are accepted only when the full objective
   improves. Because every marker retains its exact called/missing allele
   multiset, this pass may trade a decrease in optimized genotype fit against
   fewer sample diplotype changes. No separate genotype-loss bound is imposed
   on these phase-only moves; the existing full objective controls the trade-off.
   The separate genotype-fit veto is
   retained for paired exchanges inside count-reduction refits, and for the
   allele-changing macro moves and bounded interval polish below. This is a
   phase-search policy change, not imputation or calibrated phase confidence.
2. A bounded count comparison tries all K single-founder deletions. Each gets
   up to three cheap repaint/conditional-row repair sweeps, scoring the joint
   row update and the best individual update. All repaired panels compete;
   only the best repaired deletion receives deep beam/dual search and paired
   exchanges. It compares `K * complexity_cost - 2 * full_site_score`, using
   the existing complexity scale and fixed evidence mask. An optimistic local
   bound skips a reduction only if even its relaxed score loses. This is one
   deletion round, not exhaustive count selection or an assumed biological
   founder count. Cheap ranking can miss the best deeply refitted deletion.
   A second, explicitly heuristic budget screen now skips the deep refit if
   the best repaired deletion still loses by more than eight per-founder
   complexity costs. Improving repairs are never screened. This is **not**
   an optimistic bound: a deeply recoverable deletion can in principle be
   missed. Set `FounderRefinementConfig(count_refit_deficit_multiple=None)`
   to retain the unscreened search; the chosen value is checkpointed.
   Components with at most two rows or one local block bypass count search.
3. Two exact-flank window searches start from the same completed panel. One
   retains stable incumbent-suffix ranking; the other breaks exact score ties
   using a sample-relaxed future bound. Their **completed** full-site scores
   decide which trajectory survives, retaining the stable result on ties.
   A further upper-bound-ranked window pass explores beneficial tracts that
   incumbent-prefix ranking can discard prematurely. Relaxed sample-specific
   choices rank proposals only: released paths remain shared across the cohort
   and must pass canonical full-site acceptance. Optimistic bounds also prune
   windows that cannot improve the best feasible proposal. Stable/tie searches
   are reused only when all retained branch orders agree on identical inputs;
   macro searches do not use this reuse.
4. Bounded paired intervals exchange two founders over a tract, not the whole
   remaining chromosome. Local/local, L1-start/local-end and
   local-start/L1-end boundary grids retain the original emission bins.
   All pairs are considered up to seven founders. For larger panels, each
   founder nominates its three best suffix-score partners; their union has
   at most 3K pairs. This can miss a jointly beneficial interval whose suffix
   scores are weak. Both full-site score improvement and nondecreasing optimized
   genotype fit are required. These exchanges preserve the exact local
   multiset of called **and missing** alleles and cannot fill absent sequence.

The window width is 100 prepared blocks (or 100 L1 groups for the staggered
interval grid), with half-window spacing for single-founder windows. These
are bounded search budgets, not sample-count-specific biological parameters.
The existing 20-iteration and 16-branch defaults remain in use. Count deletion
can change representation; the paired exchanges alone preserve local
representation. Neither mechanism creates a new local candidate allele.

All passes have separate component, iteration and proposal checkpoints.
Intermediate component snapshots store selected prepared-row paths and
non-derived metadata rather than repeated full-chromosome arrays. Resume
reconstructs called/inference alleles, probabilities and source provenance from
the bound prepared inputs; final outputs retain the ordinary block format.
Packed scoring/suffix buffers belong to each component workspace. Independent
deletion repairs share read-only evidence in one process, keep private fit
workspaces, release the GIL in native kernels, and retain dynamic thread budgets.
Completion order never determines scientific candidate-selection order.
Release identities include the added modules and actual search configuration;
an older final result cannot silently bypass the new passes. Existing results
are retained. A changed code identity can also invalidate feedback identities
even though their mathematics is unchanged; cross-version reuse requires
explicit compatibility checks, not relabeling an old final product.
Within one code version, toggling final refinement reuses local feedback.
Local feedback still excludes final refinement, and no pass uses pedigree
information or truth. The CLI's `--founder-refinement off` disables them all.

Symmetric emissions and the uniform change penalty permit exact unordered
diploid states for this model alone. Full-site scoring costs `O(N L K²)`.
The conditional solver shares the states not involving its focal founder
across candidate choices. For t bins per block, shared preparation costs
`O(N K² t²)`, followed by `O(N (K t+t²))` per candidate. This is an exact
max-plus representation, including switches among background states. Only
retained beam paths materialize full diploid states.

Immutable background summaries are prepared once for a fixed competing panel,
then shared across focal founders, dual sweeps and overlapping windows. The
all-founder exclusion maxima require only the global best state plus separate
scans for its one/two endpoints. Forward/reverse geometries and focal state
permutations remain explicit. Caches are discarded with the panel; memory
limits select the direct equivalent kernels. Cache allocation uses one
parallel initialization pass rather than launching fills at every block.

Short full-site edits can use exact unchanged-panel flanking messages at
original block boundaries. Missing evidence retains its canonical neutral
meaning; missing founder calls are not invented. Possible winners and near
ties are canonically rescored before selection, and macro genotype-fit checks
retain the canonical full-site score and painting. A changed reference panel
invalidates its flanks. Optimistic window bounds may abort the remaining beam
only when every reachable candidate is unable to beat the best feasible
proposal; no heuristic-only rejection is added.

Across all K focal founders, the default work is cubic in K when local/macro
alphabets grow as O(K) and bin, beam and iteration budgets remain fixed. Cheap
all-deletion repairs also have cubic total work; the expensive refit is no
longer repeated K times. The linear-sized interval-pair working set prevents
a quartic interval scan. This is not a claim that chromosome length, the t²
term, or the large search constants are negligible. See the
[explicit complexity bounds](founder_scaling.md#final-founder-refinement-explicit-work-and-parallelism).
Numerical acceleration does not reduce those search budgets. Fully called
one/two-bin blocks use specialized packed beam and dual kernels; longer macro
blocks retain the general recurrence with compiled beam orchestration.
The short dual preserves the general recurrence's arithmetic order: an
algebraically equivalent simplification changed a tied path in a difficult
chromosome, so it was not retained. Beam states are neither merged nor
renormalized. Wide, ordinary beam rankings use stable partial selection.

Component-owned caches reuse packed emission addresses, invariant candidate
rankings, reverse-bin models and exact ordered-panel scores. Capped candidate
lists still retain the current incumbent in their original order. The score
cache is bounded by 64 entries and 8 MiB of keys; no process-global array cache
or stale-panel reuse is involved. Float32 full-chromosome evidence can be shared
read-only; ragged components receive contiguous gathered evidence. Batched
local emissions preserve missing-founder identity, neutral observations and
each pair's original site accumulation.

Complete-panel kernels use a site-major int8 dosage table when memory permits,
with direct scoring otherwise. Up to 64 unordered states use one uint64
traceback switch mask per marker; larger panels retain direct traceback.
These are representation changes, not phase-confidence thresholds. Independent
short windows run concurrently against the same incumbent, then replay pruning
and diagnostics in the original order before scientific selection. The dynamic
candidate allocator still redistributes released threads at kernel boundaries.
Checkpoint reconstruction preserves each probability row's original dtype.

The method remains a bounded, non-convex search. Higher read likelihood does
not guarantee fewer true founder errors. Absent local alleles, wholly unsampled
ancestry and count errors beyond the bounded one-deletion search can remain.
Greater likelihood can also trade off long-range phase accuracy. The refiner's
conservative complete-site acceptance evidence still excludes partially observed
markers, even though hierarchical linking and binned proposals use them. Those
limitations require scientific validation, not claims of a global optimum.
The [N320 fragmentation replay](validation.md#n320-fragmentation-replay) includes
an unresolved seed407 chr15 case where final refinement increases founder
errors despite successfully joined chromosome paths.

## Pedigree

Inference combines chromosome-local painting evidence with raw genotype
likelihoods, integrating evidence across physical chromosomes. The canonical
candidate-source scorer uses quadratic founder-state transitions; the fixed
top-20 parent panel keeps candidate-pair evaluation bounded per sample. The
parent-state model distinguishes zero, one and two observed parents. A missing
biological parent is not replaced by an unsupported candidate.

Ancestry-depth direction gating addresses reverse relationships and root
overcalls. It is a model assumption, not sample read-depth metadata, and may
be insufficient when generations have similar ancestry depth or parents are
missing. Exact identities, ambiguous support sets, graph adjustments, bootstrap
and leave-one-chromosome-out stability should be interpreted together. Tier B
is the primary product; the complete table is not equally supported at every
row. Fixed model priors do not require knowing the true M0/M1/M2 proportions.

The full-data ancestry-depth mixture still chooses its component count by BIC.
Chromosome bootstrap and leave-one-chromosome-out refits condition on that
selected dimension, while refitting component means, variances, weights and
sample depth posteriors. Uninformative resamples retain their neutral treatment;
a component count is never supplied from true generations or a known pedigree.
The diagnostic `AncestryDepthResampling` labels this as
`conditional_full_data_component_count`. These support fractions measure
stability **conditional on the selected model dimension**, not uncertainty
about that dimension or calibrated probabilities of correct parentage.

Real-data entry points use available design metadata for legitimate candidate
eligibility and chronology, not as individual-level trio truth. In particular,
G0/species labels do not establish parentage; sequenced outside-pedigree species
references are excluded from parent candidates. Simulation inference does not
receive the generating pedigree or generation labels.

## Family refinement and recombination

Family refinement conditions on the accepted Tier-B pedigree. Joint meiosis
messages and conditional phase polishing reconcile relatives while preserving
called genotypes and missingness. T11 has one phase-focused release policy:
start phase assessment after 20 family iterations, then require five consecutive
identical called-phase arrays, or actual family convergence. Each assessment
restarts the polisher from the current family context, not the preceding polished
path. If phase remains unstable at 520 iterations, retain that attempt and
restart from the scaffold with half damping, at most twice (0.5, then 0.25,
then 0.125 by default). Tolerance scales with damping so the undamped residual
criterion is unchanged. Successful ordinary solves are unaffected. Each attempt
has a separate checkpoint; final summaries record attempted and final damping,
and total solver iterations. Set `FamilyRefinementConfig.phase_retry_count=0`
to disable retries. If every attempt remains unstable, no final phase is released.

This is a phase point estimate, not a calibrated marginal posterior. Full family
probability tensors and a separate imputed source product are not produced.
Family-supported gap fills at the existing 0.98 context threshold can inform
the polisher internally, but do not become new published allele calls. Root
gauges and component-local labels retain their existing interpretation. T11
does not modify T10 or feed evidence upstream.

The numerical implementation retains the likelihoods, priors, error rates,
damping, root-gauge moves and coupled-branch model. Incoming messages into
hard-homozygous point masses are unnecessary; wholly fixed factors have constant
selector likelihoods initialized once. Partial, missing and soft genotype
supports retain their general calculation. Sparse copy updates preserve the
original per-marker family update order.

Ordinary selector chains integrate out neutral intervals using composed
transitions. Cached forward/backward calculations recompute changed regions
until boundary messages are exactly unchanged, without tolerance truncation.
Zero-transition phase bins are contracted exactly, preserving positive
transitions and component resets. The ordinary and branch workspaces are bounded
at approximately 24 GiB and 4 GiB respectively and never checkpointed. Gauge and
cluster changes invalidate them. These implementation reductions preserve the
model; the earlier phase-focused stopping policy is a separate scientific
trade-off validated against truth and downstream outputs.

Recombination estimation distinguishes biological switches from correlated
orientation-error tracts. Shared-family evidence can favor one parental phase
error over coincident apparent crossovers in several children. This is a
conditional model comparison, not proof: synchronized true crossovers can be
confounded with a shared error. The input phase product is not overwritten.

Maps report posterior expected crossover counts, called interval-censored
events and observable meiosis-bp exposure separately. Rates in weakly observed
regions can remain prior-sensitive; no exposure is not a measured zero rate.
Neither family inference nor the estimated map feeds back into earlier stages.
