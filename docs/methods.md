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

The default `path` selector fits a normalized within-block diploid copying
model, first to the raw discovery starts and then after each of two context
rounds. The schedule is initial local selection → L1 context/selection →
rebuilt L1+L2 context/selection → final chromosome assembly. Original latent
candidates remain available throughout. Context excludes the target block's
emission when estimating carrier weights. Its projected/refitted panels supply
candidate starts, never additional observations. The existing context refiner
can skip components above its ten-founder cap; a completed assembly round does
not imply every block received a new proposal. Local path selection still runs.

All these local fits consume the same sample- and marker-matched calibrated
genotype likelihoods and observed-read mask as discovery. Missing cells have
neutral emission one. Default calibration compares nested observation models, including a pooled
homozygote beta-binomial dispersion around the binomial-homozygote model,
alongside the existing heterozygote overdispersion, read-error and balance
parameters. Physical folds 0/1 fit parameters, fold 2 selects the model, and
fold 3 is diagnostic only; the more complex model is not forced to win. Calibration parameters are fitted from AD, not founder
count targets or pedigree truth. Production calibration sampling/refitting is
not asserted identical to private frozen-fit pilot datasets.

For a binary panel H with K rows and L kept markers, each homologue has K founder
states plus a fixed-mass unknown state. Transitions are normalized:
`T = (1-r) I + r 1 πᵀ`, where `r = 1-exp(-g ΔM)`. Genetic-map increments are
used when supplied; otherwise physical distance times the configured rate
defines ΔM. Founder frequencies are fitted; the unknown prior mass remains
fixed (default 0.01). Its alleles are independently Bernoulli(1/2) for the two
copies. Forward inference sums diploid paths, including their normalization,
rather than choosing a single fixed founder pair per sample. The objective is

`log P(GL | H,f) - log choose(2^L,K) - log K - log(K+1) + mean_k log(K f_k)`.

The last term shrinks fitted frequencies toward uniform, with total MAP
pseudocount one. This is a regularized best-panel objective, not marginal
evidence for K or an assumed ancestral founder count. Within-block mosaics need
not correspond one-to-one with the ancestral haplotypes.

Each local search uses three outer rounds, screens drop/merge/add starts, and
refits the top eight per move kind for at most twenty fixed-K updates. Ordinary
search and frequency/allele refits use the same objective. Previous-round
incumbents are protected against a worse final objective. The existing BIC
candidate-bank search and cavity-ranked source endpoints supply starts and
extra candidate rows; their scores do not replace the path objective. Partial
candidate rows use the existing nearest-latent completion while preserving
known calls. Completed latent alleles are not automatically released as calls.

Release combines a conditional one-bit probability (other panel alleles and
fitted parameters held fixed) with fractional posterior-carrier directional
support. This is not a full allele posterior or calibrated correctness
probability. Unsupported alleles remain unknown. Stored ALT probability,
probability of the current panel allele, and directional support are distinct
quantities. The posterior expected unknown-copy fraction is also separate from
the maximum observed-marker MAP unknown-copy count used by legacy completion;
neither invents a constant sample founder pair.

Optional segment exchange is off by default. After both feedback rounds, one
fixed-K pass swaps reciprocal suffixes at sixteen evenly spaced kept-marker
cuts for every row pair. Prefix frequencies stay attached to their rows;
canonicalization moves rows and frequencies together. Duplicate-producing and
no-op states are excluded. All remaining candidates receive the same normalized
score; the top eight and an ordinary warm-start control receive twenty-update
refits, protected by the original incumbent. A 1e-6 objective tie prefers the
warm control, then screening order. Exact batched screening reuses prefix
messages and pair-specific backward messages with the correctly permuted
unequal-frequency prior; the unknown state is not permuted. Neither lower
founder count nor fewer switches determines acceptance.

The explicit `balanced` and `strict` alternatives retain the earlier
cavity/BIC rescue workflow rather than being reinterpreted as path modes.
Balanced combines a cavity-ranked backbone with BIC-supported additions not
explainable by one donor join, followed by same-K cavity-ranked refinement.
Strict limits rescue to private alleles and protects surviving backbone calls.
Their bounded empty-row reduction compares smaller panels under the existing
BIC-like score and rebuilds assignments/support/calls together; partially called
rows do not trigger that reduction. Novelty filtering is a heuristic, not proof
of a distinct biological founder, and a final unknown row is not converted into
a zero-founder panel.

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
Founder refinement retains a separate fixed complete-site primary objective.
A secondary full-site partial-founder score resolves exact primary ties;
partial evidence never licenses a decrease of the primary score.

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

After each executed level of final L1–L4 assembly, the hierarchy's current
components and founder counts initialize the progressive founder refiner.
Its refined rows feed the next level. A separate bounded count-up pass runs
once on the final components. Each pass reopens row choices from the
same original **prepared local panels** within each component. A locally
supported founder can otherwise be pruned at L2 and remain
unrecoverable to L3/L4, even when a better chromosome path exists in those local
panels. Refinement changes whole local-row selections, not their allele values,
support metadata, or pre-fill inference snapshots. It neither crosses component
breaks nor adds founders, and runs only for final assembly (`max_level=4`), not
the L1/L2 feedback contexts. An irreducible hierarchy retains its existing
early stop rather than running unused levels solely to repeat refinement.

The proposal-bin minimum at L1/L2 is shared across the chromosome:
`max(configured_minimum, ceil(chromosome_sites / proposal_max_bins))`.
This avoids allocating a whole chromosome's proposal resolution to every small
component. L3/L4 retain component-specific resolution. It is a search-resolution
approximation, not mathematically equivalent to the finer early search.
Acceptance still scans individual SNPs; the 20-iteration, 16-branch and existing
count-search budgets are retained.

This pass uses the full cohort's genotype likelihoods, without truth, pedigree
or generation labels. It retains the panel scorer's normalized likelihoods,
1% uniform mixture and -2 log-likelihood floor. Its uniform cost for a change
of sample diplotype is **20**, independent of component length, at every
progressive refinement level and in count-refit/exchange passes. The hierarchy's
binned panel scorer retains its separate length-scaled penalty; the earlier
L1/L2 feedback rounds do not run this refiner. Unlike that binned scorer, acceptance
permits sample state changes at every SNP. This is a finer-discretization model
change, not merely a faster evaluation of the binned model. It is an internal
assembly fitting HMM, not the homologue-specific painting painter, a posterior phase
confidence, or a recombination-map estimator.

Fixed-painting row edits provide cheap proposals. Conditional beams vary one
founder while retaining every sample's diploid state against the other founders;
an incumbent suffix supplies complete-path ranking in both scan directions.
The first beam sees the original assembly and competes with warm proposals,
rather than inheriting a potentially worse warm-start search basin. Every
accepted fixed-count path edit improves the primary full-site score or,
at an **exact** primary tie, improves the secondary score described below.

If fine-scale proposals stall, both original and refined paths from the
**final assembly's L1 level** supply larger competing moves alongside the
incumbent path pieces. These are
not a replacement of the local panels or another feedback round. Exact duplicate
row sequences are removed; a bounded beam joins these pieces, then decodes the
proposal back to original prepared rows. All original emission bins, sample
state transitions and evidence masks remain unchanged. Accepted larger moves
are followed by the ordinary fine-scale refinement.

Larger moves have an additional genotype-fit guard. Write the optimized sample
score as `Q = E - penalty * S`, where E is the cohort's centered genotype-emission
score and S counts internal sample diplotype changes. A macro move must improve
Q, or improve partial evidence at an exact Q tie, and must not lower E beyond
numerical tolerance. This prevents accepting a
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

### Partial evidence at primary ties

The primary mask retains markers called in every original inference row.
Consequently, two different local rows can have identical primary emissions
while differing at well-observed partial-founder markers. Keeping whichever
panel arrived first can discard real evidence. Fixed-count beam, dual and
window proposals therefore use a lexicographic rule: primary score first,
then full-site partial-founder predictive score only when primary scores are
exactly equal. Near ties and small primary losses do not qualify.

The secondary score uses the same seven genotype distributions, shared-unknown
source-row identities, 1% mixture and likelihood floor as the proposal model.
It permits sample-state changes at every SNP. Its mask is fixed from the original
prepared inputs: keep flags plus at least one called founder. All-founder-unknown
sites and unobserved sample evidence remain neutral. This is the existing
per-observation predictive approximation, not joint latent-allele integration,
a calibrated phase posterior, or new allele imputation. Count comparisons and
the primary genotype-fit guard are unchanged.

An unrestricted partial-data optimum can trade away primary fit elsewhere
and consequently be rejected. When the ordinary dual/macro search stalls,
a bounded extra dual search groups local rows by their **identical alleles at
every primary-scored marker**. Each focal path may explore only its incumbent's
equivalence class, with the same branch cap and sweep budget. Every decoded
path must preserve the canonical primary score exactly and improve the full-site
secondary score to be accepted. This reuses the existing solver and does not
enumerate all local edits with a whole-chromosome refit for each.

### Remaining passes and implementation

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
   decide which trajectory survives. Exact primary ties use the secondary
   partial-founder score; a tie in both scores retains the stable result.
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

After the final executed hierarchy level, symmetric count-up/refit is enabled
by default. It tries a duplicated-incumbent and an unused-local-row start in
both directions, retains the best canonical proposal, then repairs the whole
K+1 panel with the same fixed-count primal/dual and paired passes given to the
incumbent. The same complexity objective decides acceptance. At most two
additions are tried, stopping at the first rejection. It neither assumes K nor
invents local alleles. Single-local-block components are left to discovery.
This is a bounded search, not exhaustive founder-count estimation. Set
`FounderRefinementConfig(count_max_additions=0)` to disable additions only.

The window width is 100 prepared blocks (or 100 L1 groups for the staggered
interval grid), with half-window spacing for single-founder windows. These
are bounded search budgets, not sample-count-specific biological parameters.
The existing 20-iteration and 16-branch defaults remain in use. Count deletion
can change representation; the paired exchanges alone preserve local
representation. Neither mechanism creates a new local candidate allele.

Each final hierarchy level has a separate refinement checkpoint namespace.
Independent components run concurrently within the shared core and memory
budget; each worker retains its own fitting workspace. Small components save
compact completed results rather than thousands of tiny proposal files.
Long components retain separate iteration and proposal checkpoints.
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

The genetic score averages two forward log scores with equal weight:
`s(H) = 0.5 * log Z_eta(H) + 0.5 * log Z_1(H)`, for each M0/M1/M2
configuration `H`. Both use the same selected markers, genotype likelihoods,
painting-derived candidate sources, and transmission transitions. In
`Z_eta`, the existing child/bin information exponent tempers emissions
**before** summing over transmission paths; `Z_1` uses full-strength
emissions. Tempering before path marginalization can suppress the
contradictions that separate a missing parent from a related observed
candidate. The second view preserves that information; retaining the first
view improved robustness in the noisy-read control.

This is a composite score over alternative evidence-weighting assumptions,
not independent replicated data or an exact calibrated likelihood. The
equal weight is fixed, not selected per fish or fitted using pedigree truth.
Both candidate screening and final M0/M1/M2 scoring use the same pool.
Wholly missing child observations remain neutral; an entirely uninformative
candidate adds no second-parent evidence. Projected sources and pooled M0/M1
screen scores are reused, preserving quadratic founder-state scaling.
The score recipe has its own checkpoint identity: existing preparation can
be reused, but old genetic scores are not silently treated as the new model.

Default pooled predictive calibration (`--pedigree-calibration predictive`;
disable with `--pedigree-calibration off`) multiplies all M0/M1/M2 chromosome
log-likelihoods by one positive scale. Whole usable chromosomes, in input order
modulo four, form four groups.
Each of the first three groups is predicted by the other two; their summed log
predictive mixture selects the scale. The fourth group supplies untouched
diagnostic testing, predicted using all three non-test groups. Existing fixed
state priors and eligible candidate counts are preserved; cohort prevalence is
not fitted. A small log-scale grid brackets numerical maxima before continuous
refinement because this objective need not be unimodal. Search bounds are
0.001–100; boundary/nonconverged fits or no selection gain retain scale 1, as do
inputs with fewer than two chromosomes in any group.

This is conditional predictive weighting of a composite model, **not** calibration
of biological posterior probabilities. The paintings and candidate panel were
constructed using all chromosomes, and relatives are dependent. Direction,
reciprocal-family inference and release retain their existing assumptions.
Bootstrap/LOCO refits condition on this fitted scale, so they do not include its
estimation uncertainty. The untouched test gain is recorded, never used to tune
the weight. Full-marker Mendelian exclusion uses the original raw genotype
likelihoods and recomputes any newly required candidate dyads without this scale.

Pedigree uses [finite direction and family evidence](pedigree_direction.md) rather
than a mandatory ordering of inferred ancestry layers. Paired chromosome
junction contrasts give finite, neutral-centred orientation support; four
synchronous reciprocal-family cavity-message passes compare competing M0/M1/M2
families while excluding immediate reverse feedback. A joint-family
factor also checks reverse two-edge ancestry paths, averaging over uncertain
parent configurations and counting shared intermediates once. Its top-16
configuration panel assigns omitted mass neutral compatibility; conditioning on
both focal parents preserves legitimate backcrosses. Freeze this path evidence,
then replace the initial reciprocal term with a converged reciprocal
solve on the path-adjusted scores (maximum log-message change `1e-8`, at most
128 undamped iterations, then a 4096-iteration damping-0.5 restart if needed;
the residual is measured before damping and failure of both attempts is reported). The final messages do not recompute the
supporting path factor. Explicit chronology
exempts that parental side without exempting an uncertain co-parent. All these
terms are recomputed during bootstrap and leave-one-chromosome-out fits.
Exposure requirements, explicit
eligibility, the fixed top-20 pair panel, fixed state priors and release
thresholds remain unchanged for chromosome-resampling release.
The standard workflow additionally scans full-marker raw genotype likelihoods
for unresolved/non-exact M0/M1 counts. Mendelian exclusions and already-supported
outgoing edges must jointly leave exactly the full-data incoming parent set;
no new edge is introduced. This fixed-observed-genome release is labelled
separately and preserves original resampling diagnostics. Its error bound
requires calibrated likelihoods/independent read errors; directional release
also depends on supported directions being correct. It is not an unconditional
pedigree-accuracy guarantee. See the direction document for the bound and outputs.

This is a bounded composite/loopy approximation, not exact global pedigree
marginalization. Callability correction uses chromosome summaries rather than
exact common-interval counts. Tier B remains the primary product, and internal
support is not a calibrated correctness probability. M0 means zero observed
parents, not that the fish cannot have observed descendants. The prior
cluster policy remains available for controlled API comparisons; it is not the
workflow default.

Real-data entry points use available design metadata for legitimate candidate
eligibility and chronology, not as individual-level trio truth. In particular,
G0/species labels do not establish parentage; sequenced outside-pedigree species
references are excluded from parent candidates. Simulation inference does not
receive the generating pedigree or generation labels.

## Family refinement and recombination

Family refinement conditions on explicitly supported Tier-B parent edges.
The exact configuration table remains the primary pedigree report. Separate
`tier_b_partial_relationships` rows retain marginally supported edges even
when M1 versus M2 is unresolved: an edge must meet the existing bootstrap and
LOCO thresholds in both local and graph views and belong to the full-data
acyclic graph. The table records support flags and lower/upper observed-parent
counts; an unresolved co-parent is not guessed. Family refinement and
recombination use the same partial table, and their relationship checkpoint
identity includes the support flags. Joint meiosis
messages and conditional phase polishing reconcile relatives while preserving
called genotypes and missingness. Family phase has one phase-focused release policy:
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
gauges and component-local labels retain their existing interpretation. Family phase
does not modify pedigree or feed evidence upstream.

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
