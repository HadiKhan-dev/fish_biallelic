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
See [configuration and checkpointing](running.md#local-feedback-selection) and
[local validation](validation.md#local-feedback-selection).

## Hierarchical linking

All four assembly levels use the same distance-aware normal/error-tract linker
in `assembly/linking.py`, with at most 20 EM iterations per block gap and the
same convergence rule. The linker uses the input genetic map where supplied,
or the configured scalar rate otherwise. L1 keeps its existing block grouping
and unlimited beam-gap rule; higher levels retain their distance-based beam-gap
limit. Missing-data component boundaries and founder-source provenance are
preserved at every level. Large linking proxies are sampled from markers called
in every frozen founder candidate; sample eligibility and boundary support are
checked on those actual proxies. No usable evidence means unresolved components.

The implementation stores three genotype emissions per sample/site, reuses
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
unclipped mixture emissions and iteration cap. Incompatible assemblies are not
silently reused; changing the linker requires new assembly/downstream identities,
not regeneration of the underlying simulated reads or discovered blocks.

## Final founder-path refinement

The hierarchy chooses component boundaries and founder counts. Its final pass
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

The default starts with width64, widens fourfold up to 1024 when the best beam
improves on the cheaper proposal by more than one switch penalty, and retains
that budget for subsequent sweeps. This is a computational heuristic, not a
confidence threshold. There are at most20 sweeps, one focal path per sweep and
16 candidate local rows per beam expansion. Unknown observations are neutral;
the evidence mask is fixed across all original local candidate rows, so changing
a path cannot improve its score by hiding difficult sites. Original missing
calls can still move with the selected row.

Symmetric emissions and the uniform change penalty permit exact unordered
diploid states for this model alone. Full-site scoring costs `O(N L K²)`;
conditional beam work costs `O(N M W C K²)` per sweep, where M is the total
proposal-bin count, W the bounded beam width and C the bounded local branch
count. Traceback bookkeeping is linear in original block count. This does not
introduce cubic or quartic founder-state transitions into structured assembly.
Macro emission packing adds `O(N M (K+R)²)` work/storage, with R the number of
L1 context paths per piece. The additional genotype-fit check uses the same
quadratic-state traceback only for score-improving macro proposals. There is no
new cubic or quartic transition, and the original search budgets are retained.

The method remains a bounded, non-convex search. Higher read likelihood does
not guarantee fewer true founder errors; an absent local allele or incorrect
final founder count cannot be repaired by this fixed-count pass. Conservative
complete-site evidence can exclude useful partially observed markers. Those
limitations require scientific validation, not claims of a global optimum.

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
path. If phase remains unstable at 520 iterations, retain work and refuse release.

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
