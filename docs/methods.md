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

Painting release. The painter publishes, per sample and bin, the most specific
statement whose posterior reaches 0.9 (`minimum_viterbi_public_class_posterior`).
First the unordered pair of founder classes of the Viterbi state (qV), as
before. Where two kept founder rows are identical over a bin's selected sites
(same {-1,0,1} calls), the reads cannot separate them and the founder label is
not identified; then the unordered pair of per-bin allele-equivalence classes
(qA, the summed posterior of their member rows) is released if qA >= 0.9 and
the sample has an observed, non-uniform read at a site of the bin where the
pooled class is called. Such a track has status `LOCAL_ALLELE_EQUIVALENCE`,
label -1 and known alleles; its chunk names the candidate rows
(`hap1_members`/`hap2_members`, sorted component-local rows; one row for a
named track, none for a withheld one) and chunks split where a member set
changes. Without reads in the bin the pair is withheld, as is everything below
0.9. Where the two track orders of the decoded path have equal HMM score, the
order with fewer allele-visible changes is kept (a change between identical
rows is invisible). The HMM, its threshold and every statement released by the
label rule are unchanged; a label change between identical rows inside a
pooled run is not placed at any bin.

## Site quality

On VCF/BCF input, each contig's loaded records pass a self-calibrated filter
before read calibration and discovery (`core/site_quality.py`). It uses only
allele depths and record type, never labels, benchmarks or truth. A record is
removed when it is not a biallelic SNP (r0); its total depth lies outside the
clean core of the contig's total-depth histogram (r1); its per-sample depths
are more heterogeneous than the clean core allows given the site's own depth
(r2); a negative-binomial artefact class explains its wrong-allele reads in
homozygous-looking samples better than the Poisson clean core (r3); a
beta-binomial artefact class explains its heterozygote allele balance better
than the binomial clean core (r4); or it has fewer than two expected carriers
of a non-major genotype (r5, informativeness rather than quality). A clean
core is the histogram mode with widths from its half-maximum crossings; its
window is where the core's local posterior is at least one half. A mixture
whose artefact class does not separate from the clean core removes nothing.
Genotypes are classified under a fixed read model (error 0.01, balance 0.5);
the clean-core rates are then fitted. Kept sites are re-blocked 200 at a time,
exactly as the loader blocks an input restricted to them. The filter's
parameters and per-rule counts are in the discovery checkpoint and run record.

The filter is off by default for simulations. Depth rules r1/r2 cannot tell
benign sample-specific missingness (the stress control's dropout tracts) from
mapping artefacts; when the read error rate is several times the 1%
classification error (stress control: 4%), homozygotes with error reads are
classified heterozygous and r4 separates spuriously; r5 removes
founder-monomorphic simulated sites. Evaluation compares final phase with truth
on the full simulated site axis.

## Sample quality

Inside the site filter, each contig also checks every analysed sample for
library contamination (`core/sample_quality.py`). Foreign DNA at read fraction
alpha < 1/2 shows alleles that no other sample carries, on every contig, at a
minor-read fraction near alpha (foreign homozygote) or alpha/2 (foreign
heterozygote). A sample's own heterozygous allele that no other sample carries
(an inherited segment absent from the sequenced founders, or a founder segment
no sequenced descendant received) shows the heterozygote balance instead,
about one half. Only the first is flagged.

At biallelic SNPs that pass r0-r4 above (whole analysed cohort), the
cohort-minor allele is ALT when the cohort's ALT reads do not exceed its REF
reads. A sample's private cells are those where it alone has a read of that
allele; private cells of one sample less than 1 kb apart can share sequenced
fragments and count as one event. The minor reads of each private cell are
modelled as a mixture of a heterozygous allele (fraction h, the contig's r4
clean heterozygote balance, or 1 - h for a REF minor allele) and one
low-fraction allele (fraction fitted per sample in (0, h/2], h/2 being the
largest fraction of a foreign heterozygous allele while foreign DNA is under
half the library), both binomial and truncated at one read; the low-fraction
events are that component's event-weighted posterior mass at the sample's
maximum-likelihood fit. The statistic is the square root (the
variance-stabilising scale of event counts) of low-fraction events per
observed cell, and a sample is flagged on the contig when its robust z against
the cohort (median and 1.4826 x MAD on that contig, the spread no smaller than
the statistic's Poisson counting noise, 0.5 / sqrt(observed cells)) exceeds 5.
No labels, pedigree, benchmark or absolute rate enter; cohorts under 20
samples flag nothing. The decision is per contig, so a contig's result
does not depend on which other contigs a run includes. On real TroMau data
(22 autosomes) one library is flagged on every contig (z 20-41, fitted low
fraction 0.19-0.25); no other library exceeds z 4 on any contig, and none
would be flagged by the same statistic pooled over the 22 contigs.

Heterozygous near-private alleles (the sample and at most one partner carry
them) never flag a sample. They are reported per contig in
`<output>/sample_quality/<contig>.segments.tsv`: cells whose reads favour the
heterozygous fraction over h/2 are grouped per (sample, partner) into fragment
events and into regions of events less than 1 Mb apart; regions with at least
three such events are listed for unflagged samples with the region's cell
count and the sample's and partner's minor-read fractions (on TroMau, samples
without such segments never chain more than two events). On TroMau these are 1-10
Mb segments shared by pairs of relatives or private to one sample, including
a G0 segment carried by a single F1. They explain added founder rows and pedigree
misfits; they are not evidence of contamination.

On the contig where it is flagged, a sample's reads are excluded from founder
construction: block discovery sees it without reads, and feedback, assembly
and founder refinement see its cells as unobserved (uniform and unobserved,
like zero-depth cells), as does the founder allele refit after painting; r5
counts carriers among unflagged samples only. Read calibration, painting, the
pedigree, family phase and recombination keep every sample's full evidence, so
flagged samples are still painted, phased and scored. The assembly release
identity hashes the founder evidence, so the stages that check their raw
evidence against it (pedigree, allele refit, family phase) apply the recorded
flags for that check. Per-sample statistics, z and flags are in
`<output>/sample_quality/<contig>.tsv`, the run record and the chromosome's
`discovery/prepared` record (`site_quality["sample_quality"]`,
`founder_evidence_excluded`).
Contamination shared by several libraries from one source is not private and
is seen only in part, and DNA from another analysed sample brings no private
alleles, so contamination within the cohort is not detected. The QC runs only
where the site filter is applied, so it is off in simulations by default.

## Local feedback and candidate selection

The default `path` selector fits a normalized within-block diploid copying
model, first to the raw discovery starts and then after each of two context
rounds. The schedule is initial local selection → L1 context/selection →
rebuilt L1+L2 context/selection → final chromosome assembly. Original latent
candidates remain available throughout. Context excludes the target block's
emission when estimating carrier weights. Its projected/refitted panels supply
candidate starts, never additional observations. Every completely covered
context component is refitted; partially covered contexts keep their input
panels, so a completed assembly round does not imply every block received a new
proposal. Local path selection still runs.

The refit's site posterior over the K shared context-founder alleles is
p(b) ∝ ∏_n Q_n(b), where each informative sample contributes a positive
quadratic Q_n (its cavity pair weights times its genotype evidence, the unknown
state's alleles integrated per copy). For K ≤ 10 every one of the 2^K
configurations is enumerated. Above that (`assembly/site_posterior.py`), the
posterior mode is found by loopy belief propagation over the sample factors and
single/pair-flip ascent, started from the propagation's thresholded beliefs and
from their full allele complement; the better optimum is kept. When the two
starts end in different optima within 20 nats of each other, the posterior has
two comparable basins that no single or pair flip connects, the local
certainty argument below does not hold, and the site releases nothing
("multimodal" abstention). This happens where most carriers look heterozygous,
for example at paralogous sites: the posterior is then nearly invariant under
complementing every founder allele, or frustrated like a maximum cut. The
founder-allele model is misspecified there and the exact enumeration is itself
unreliable (on constructed paralog-like sites 35-49% of its releases contradict
the simulated founders); the abstention removes most, not all, releases at such
sites when the two starts happen to agree. Otherwise founders whose single
flip and every pair flip cost more than 20 nats stay at the mode, and the
remaining core is enumerated exactly when it has at most ten founders,
otherwise over the recentred Hamming ball of the largest radius with at most
1,024 configurations but at least 3 (up to 4,526 configurations for a
30-founder core). In that ball, a release is withheld when the marginal lies
within three times the weight of the outermost-shell configurations that flip
the founder of the 0.99 threshold.
Exact marginals of this model are #P-hard in general, so above ten founders the
marginals are exact only over the enumerated support; the work is O(N K³) per
site. The same rule, including the multimodal abstention, replaces the 2^U
enumeration of the assembly completion (joint completion and whole-bin cavity
fill) above ten unresolved founders at a site; the joint completion's EM keeps
the selected support fixed, so its ELBO remains a lower bound.

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

`log P(GL | H,f) - log choose(2^L,U) - log K - log(K+1) + mean_k log(K f_k)`.

The last term shrinks fitted frequencies toward uniform, with total MAP
pseudocount one. U counts the rows coded as new arbitrary haplotypes. In the
initial fit U = K. In the feedback rounds the assembled context names its
founders in the block (each founder's cavity-weighted refit, the feedback
proposal); a founder's local sequence in H is the row nearest its proposal on
the proposal's called sites, and a row some founder claims is a parameter of a
founder whose existence the multi-block assembly decided, not a new haplotype:
U counts only the unclaimed rows (zero code when U = 0). Rows beyond the
context's founders (local recombinants, noise rows) keep the full code; blocks
no context component covers keep U = K. The fixed-K fit is unaffected (its
dictionary term is constant). This is a regularized best-panel objective, not
marginal evidence for K or an assumed ancestral founder count. Within-block
mosaics need not correspond one-to-one with the ancestral haplotypes.

Each local search runs rounds until the best refit no longer improves. A round
screens every drop and add start by its exact forward score and every merge
start by an EM lower bound: the merged panel is the drop panel of either row
with the other row replaced by the consensus, so the drop panel's posterior
gives LL(drop) + the replaced row's complete-data flip gains at the changed
sites + the expected-copy frequency term, and the larger of the two bounds
ranks the merge. The top eight starts per move kind are refitted to a
conditional fixed point. The limits 10 + 2K rounds and 20 + 4K fixed-K updates
are reported backstops, as are the candidate-bank limits (256 + 16K scores,
128 + 8K starts per round, 8 + K/2 rounds). Ordinary search and
frequency/allele refits use the same objective. Previous-round
incumbents are protected against a worse final objective. The existing BIC
candidate-bank search and cavity-ranked source endpoints supply starts and
extra candidate rows; their scores do not replace the path objective. Partial
candidate rows use the existing nearest-latent completion while preserving
known calls. Completed latent alleles are not automatically released as calls.
At L1 and L2 the context start is fitted first; when the previous-round
incumbent has a higher objective, the search refits and explores from the
incumbent instead, so it does not climb back to it. Otherwise the search is
unchanged. A reused search returns the previous result, as before.

The local forward/backward implementation exploits exchange symmetry between
the two homologues, including when founder frequencies are unequal. It still
sums all ordered-state mass: off-diagonal cells count twice. This is an exact
property of the current unphased-GL model, not frequency tying or state pruning.
Predicted trajectories store only the upper triangle, and the two identical
filtered marginals share one array. Backward reductions are fused; conditional
flip calculations retain positive noncarrier sums rather than subtracting two
nearly equal totals. No fastmath, iteration-budget or likelihood change is used.
Completed searches can be reused across feedback rounds only when the initial
panel, ordered candidate bank, observations, marker/map coordinates and all
search/model settings match. Their release calculation is reusable only with
the same calling thresholds; cached posteriors never become new observations.
Within one candidate-bank call, identical panels and sample assignments also
share release calculations. The BIC bank computes cavity scores for its source
endpoints, where they determine the starting panel, but omits unused top-bank
cavity diagnostics. Uncomputed fields are `None`, not zero or confidence
estimates. Explicit cavity selection still scores every admitted mode.

Release combines a conditional one-bit probability (other panel alleles and
fitted parameters held fixed) with carrier support among the samples whose own
evidence favours the allele: at least two of them each more likely than not
(posterior carrier probability above 1/2) carry the founder at the site, or
their expected carrier mass is at least two. This is not a full allele
posterior or calibrated correctness probability. Unsupported alleles remain
unknown. Stored ALT probability, probability of the current panel allele, the
supporter count and the expected carrier mass are distinct quantities. Every
founder-allele release uses the same carrier support over its own sample set:
the feedback refit (q >= 0.99), the joint completion's cross-fitted release and
the whole-bin cavity fill (each q >= 0.99). The mass alone sits just below two
for a founder with exactly two near-certain carriers, so it in effect required
three; the count alone withholds diffusely supported sites that the mass
releases (on one simulation that withholding flipped a fragile chromosome-level
join into a chimera), so release needs either. The posterior expected
unknown-copy fraction is also separate from the maximum observed-marker MAP
unknown-copy count used by legacy completion; neither invents a constant
sample founder pair.

Path selection is the only feedback selection. The former balanced/strict
cavity/BIC rescue selections and the optional segment exchange can no longer be
selected.

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
the dense linker at every hierarchy level. Assembly uses a certified
candidate-panel search under the join likelihood (per sample, the best sequence
of unordered panel-path pairs over binned emissions, minus a uniform penalty per
state change).

The search starts from the state-covering initial panel of largest likelihood
among those of every input anchor. Same-count edits (replacements, one-block
row changes, suffix copies and exchanges, cycles) are accepted when they
strictly improve the likelihood and leave every changed path a private cell
and more than 1% of called sites away from every other path. At each stall the
count step decides one founder-count change with the pipeline's
effective-carrier rule (as founder refinement, cost c per founder charged to
the effective carriers, leave-one-sample-out check, bootstrap band), applied to
the per-sample total gain and to its sequence part: the gain left when the
decided path's private cells are replaced by the smaller panel's nearest rows.
Switch savings alone therefore never justify a path; a path that is a block
mosaic of the other paths, or near-identical to one over the span, is deleted.
Deleting a path that a birth of the same join added is a count cycle: the rule,
called against different panels, has both added and deleted that row. The
deletion stands and births then stop. A cycle is recorded as undecided when its
closing deletion is ambiguous (bootstrap band containing zero) and it began with
a decided deletion (not with the birth, and not with the cleanup deletion of a
path without sequence of its own); such calls are published in
`founder_counts.tsv`.
A founder that is a pure block mosaic of other founders within one batch is not
carried by that batch (no pedigree information is used); later levels and
founder refinement can re-split it. Founder refinement decides its counts with
the same rule, with the sequence part always scored exactly (no near-identity
shortcut). No biological founder count is supplied. A tier-2 pass
certifies every same-count move of an explicit neighbourhood once the count
step is quiescent; certification is relative to that neighbourhood and the
endpoint-beam pool, not a global optimum.

Dense propagation is cubic in founder count (see
[founder scaling](founder_scaling.md)). Numerically guarded matrix multiplication
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
The initial local refit's identity is independent of subsequent assembly
settings, so its unchanged selected panels and partial block fits can be reused.
Local implementation hashes and input/model identities must still match:
assembly-independent identity does not make older local fits compatible with
changed fitting code. Original discovery checkpoints remain preserved.
L1/L2 feedback contexts, later selections, and final assembly still require
new identities when the panel-search algorithm changes.

For bounded-panel score-only Viterbi, exchange-symmetric diplotypes are stored
once as unordered pairs; the maximum is unchanged and ordered traceback retains
its original tie choices. Birth proposals score each distinct local row once,
then restore the original chromosome-candidate order. Fully called binned
emissions evaluate the three genotype likelihoods once per sample/site and
reuse them across pairs. Partial-founder emissions retain their existing model.
These implementation changes do not reduce proposal budgets or alter acceptance.

## Final founder-path refinement

After each executed level of **final** L1–L4 assembly, the hierarchy's current
components initialize the founder refiner; its output feeds the next level.
It never runs in the L1/L2 local-feedback contexts. Original prepared local rows,
their allele values and evidence masks stay fixed. Refinement changes their
chromosome-wide arrangement and decides the founder count: earlier levels may
only delete rows, and the last level (the final count step) may also add rows.
No target founder count, pedigree or simulation truth is supplied. This is the
only refinement behaviour: no option selects another search, count criterion,
panel rule or count-cycle rule.

### Objective

The canonical objective F is the full-site cohort score of
`assembly/founder/scoring.py` over panels of original prepared local rows.
The acceptance HMM retains normalized genotype likelihoods, the 1% uniform
mixture, -2 log-likelihood floor, and a cost of **20** per sample diplotype
change at every level. It allows changes at every SNP. This internal
unordered-diplotype assembly model is not the downstream homologue painter,
a phase posterior, a genealogy model, or a recombination-map estimate.
Its mask is fixed across the original candidate rows, so a path cannot improve
its score by hiding difficult markers.

### Converged fixed-count fits

A fixed-K fit repeats Gauss-Seidel sweeps until a sweep accepts nothing
(`assembly/founder/converged_sweep.py`). Each sweep computes exact forward and
backward messages of the panel and, from the panel at the start of the sweep,
proposals: for every founder and every 8-block window (stride 4) the exact
maximiser of the binned window problem and, where an L1 context exists, the
same over windows of four consecutive L1 spans (stride 2), offering the
founders' current span tuples plus the original and refined L1 paths. One exact
pass in block order, alternating direction between sweeps, then applies at every
junction the best founder-pair suffix exchange while its exact gain exceeds a
small tolerance, and re-scores the proposals anchored there exactly against the
current panel, best first. The canonical F is recomputed and must equal the
start score plus the accepted gains.

Every fixed-K fit runs this search twice from the same start, concurrently: once
with its first pass forward and once with its first pass reversed (both then
alternate). Both are deterministic local searches; the fit keeps the one with
the higher exact F (exact ties: forward-first), so at a fixed start it never
ends below the forward-first search, and both results are reported
(`best_of_two` in the fit records). This applies to level start fits and to
every count-decision refit.

A converged fit is a certified local optimum of F under this move set: every
proposal of its last sweep was computed against the final panel and every move
was scored exactly against it. F is non-convex; no global optimum is claimed.
Window proposals optimize a binned surrogate (bins of at least 200 SNPs,
targeting 2,000 bins per component; L1/L2 share a chromosome-derived minimum),
so the certificate is relative to them; acceptance always uses individual SNPs.
A fit reaching the 100-sweep backstop is marked non-converged in its
diagnostics, never silently.

Coarse (L1-span) moves pass a guard. A long founder edit can raise F by letting
several samples drop the switch of a crossover they share, turning the founder
into an artificial recombinant. The default `shared_junction` guard refuses a
move that lowers the genotype fit E = F + 20 x switches only when its F gain
does not exceed the tolerance once every saved switch at a one-way junction
shared by two or more samples is uncredited. This can withhold a true move and is
not a confidence threshold or a proof of biological correctness.

### Founder-count decisions

Count decisions compare converged fits one row apart, at most r = 9 converged
refits per decision (`COUNT_RANKED_REFITS`). The candidates of a deletion round
are the rows without a private cell (every block row is used by another row;
their sequence part is zero, so the rule deletes whichever is decided) when there
are any, as in the join, and otherwise every row. A deletion round refits every candidate's single-row
deletion of the incumbent to convergence when there are at most r candidates;
otherwise the deletions are first scored exactly without refitting (per-sample
canonical scores) and ranked by the criterion's point margin on those scores
(below: the smaller of m and its leave-one-sample-out version, on the total
gain), and the r top-ranked are refitted. The criterion then decides between the best-F refit
(exact ties: the lower row) and the incumbent; a rejection ends the deletions
(backward elimination with refit). An addition round (final count step only)
refits every duplicated-row seed, or, above r rows, the r seeds whose copy gains
most from its exact best responses on all 8-block windows (one copy-only pass of
exact gains), and decides on the best-F of them. The refits of a round run
concurrently, each with its share of the threads. With more than r candidates the
ranking chooses which are refitted, so it can miss a deletion or addition that only
a refit of another candidate would show. An optimistic bound may skip a round
only when it proves the decision for every candidate; an addition round also when
no sample's gain under any panel with one row more can exceed the criterion's
per-carrier threshold c/(2S), which keeps the point margin and every bootstrap
replicate below zero (the decision is then recorded as skipped by the bound); a deletion round only when
every row has a private cell, where the per-sample bound on any panel with one
row fewer also bounds every shadow panel (below). An accepted addition is
followed by one deletion round from the larger panel, which must reject;
otherwise the call is a count cycle. Deletions stop at two rows, and additions
when the bound excludes them, the best seed is rejected, or the count reaches
twice the incoming count (a reported backstop).

Each decision uses the effective-carrier criterion
(`assembly/founder/count_criterion.py`). With per-sample score differences d_s
between the larger and the smaller panel, the per-founder cost c of
`chimera_scoring.compute_cc` and S samples, the larger panel is kept or added
only if m = sum_s d_s / c - n_eff / (2S) > 0 and the same margin with the sample
of largest d_s removed is also positive; n_eff is the participation ratio of the
positive d_s. The cost is thus charged only to a founder's effective carriers,
and no single sample can decide a row. As in the join, the test must pass on the
per-sample total gain d_s and on its sequence part q_s = F_s(X+) - F_s(X+^sigma):
the shadow X+^sigma replaces every cell of the larger panel that a reference
panel does not use by the reference's nearest row at that block (Hamming over
called sites); the reference is the larger panel without the decided row for a
deletion, and the incumbent for an addition (whose refit may have moved any row).
A larger panel holding a row without a private cell has q = 0, so switch savings
alone never justify a row. Unlike the join, refinement takes no near-identity
shortcut: a row within 1% of another over the component keeps the evidence of its
differing cells, because the last level is final and a merged pair could not be
re-split later. A sample bootstrap of the smaller of the two
margins (2,000 resamples, 95% interval) marks calls whose interval contains zero
as ambiguous; they keep the point decision. A deletion round that keeps the
incumbent reports the other rows whose sequence part fails the test
(diagnostics only: only the decided row is tested, as in the join).

The count decisions choose K. The panel kept at that K is the best-F converged
fit computed at K during the count pass; it differs from the last panel only when
the pass returned to a K it had held with a better panel. A count cycle is
resolved to the smaller K (the deletion round stands) with the better-F of its
two K panels, and is reported as ambiguous; it is undecided, as in the join,
when the deletion round's call is ambiguous and the pass reached K by a decided
deletion from K + 1. When a best-F panel replaces the last
one, the diagnostics record the replacement and the restored panel's adjacent-K
comparisons: its recorded decisions and its criterion margin,
leave-one-sample-out margin and bootstrap band against the best-F fits computed
at K-1 and K+1. The call is flagged ambiguous when the restored panel fails its
own deletion test. Best-F is a two-stage policy (the count decisions choose K,
the fixed-K objective chooses the panel), so a restored panel can fail that
test, as on one simulated F3 sib-family chromosome.

### Implementation and limitations

Independent components of a level run in worker processes, each with its share
of the verified CPU budget (re-read at every sweep, so the level's last
components take over the cores of finished ones); window programmes and an
addition round's seed responses run concurrently inside a component, and the
acceptance pass uses the component's threads for its per-sample kernels and
evaluates chunks of junctions at once until something is accepted. Completion
order never sets scientific selection order, and results
do not depend on the thread count. Each final hierarchy level records every
refined component (`assembly/l<k>/refine/<n>`) as it completes, so an
interrupted level resumes from its completed components. Source
identities include every numerical module of the refiner.
`--founder-refinement off` disables refinement and count decisions.

On the simulation benchmarks of 5–6 October 2026, K equalled the identifiable
truth on 61 of 64 chromosomes (two splits of near-identical founders and one
founder that was never assembled). The criterion can miss rare founders carried
by two to five samples. Absent local alleles, unsampled ancestry, weak founder
representation, founders missing from the discovery candidates and long-range
mislinking remain limitations. More likelihood or more called alleles does not
by itself establish improved biological reconstruction.
See [scaling and work bounds](founder_scaling.md#final-founder-refinement-explicit-work-and-parallelism).

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

A candidate parent's transmitted-state chain follows its painting posterior,
whose switch penalty is a painter's regularizer, not a probability. Where the
parent has no informative selected markers, that chain is interpolation only.
Each parent bridge is therefore mixed with the uniform selector kernel,
`T' = (1-k) T + k U`, where `k` is the boundary's meiotic switch probability
scaled by `1 - min(1, n/N)`; `n` is the smaller informative-marker count of the
parent's two adjacent bins. The default `N=1` acts only next to data-free
parent bins, and `N=3` gives a graded form
(`parent_state_source_gap_support_sites`; `0` restores the unmixed chain).
M0 is unaffected, and data-supported parent bins keep their bridge unchanged.

Default pooled predictive calibration (`--pedigree-calibration predictive`;
disable with `--pedigree-calibration off`) multiplies all M0/M1/M2 chromosome
log-likelihoods by one positive scale. Whole usable chromosomes, in input order
modulo four, form four groups.
Each of the first three groups is predicted by the other two; their summed log
predictive mixture selects the scale. The fourth group supplies untouched
diagnostic testing, predicted using all three non-test groups. The objective
omits M1/M2 rows whose proposed parent `p` is reciprocal-dominant on the
fitting chromosomes: `M1(p | c) - M0(p) >= M1(c | p) - M0(c)` for child `c`.
The genetic score cannot order such parent/offspring reversals; direction and
reciprocal-family inference settle them, and decisions still see every row.
Existing fixed
state priors and eligible candidate counts are preserved; cohort prevalence is
not fitted. A small log-scale grid brackets numerical maxima before continuous
refinement because this objective need not be unimodal. Search bounds are
0.001–100; boundary/nonconverged fits, no selection gain or a fitted scale above 1
(the weight only tempers, never sharpens) retain scale 1, as do inputs with fewer
than two chromosomes in any group.

This is conditional predictive weighting of a composite model, **not** calibration
of biological posterior probabilities. The paintings and candidate panel were
constructed using all chromosomes, and relatives are dependent. Direction,
reciprocal-family inference and release retain their existing assumptions.
Bootstrap/LOCO refits condition on this fitted scale, so they do not include its
estimation uncertainty. The untouched test gain is recorded, never used to tune
the weight. Full-marker Mendelian exclusion uses the original raw genotype
likelihoods and recomputes any newly required candidate dyads without this scale.

Pedigree uses [finite direction and family evidence](pedigree_direction.md) rather
than a mandatory ordering of inferred ancestry layers. A per-chromosome count
likelihood of painted junctions (a transmitted homologue carries each parental
junction with probability 1/2; everything else is a non-negative remainder; a
5% per-chromosome painting-failure floor) gives orientation log odds that add
over chromosomes; a 2% pair-level painting-failure mixture (sample-level
junction excess repeats on every chromosome) bounds the neutral-centred
correction at 3.9 nats against an edge; four
synchronous reciprocal-family cavity-message passes compare competing M0/M1/M2
families while excluding immediate reverse feedback. A joint-family
factor also checks reverse two-edge ancestry paths, averaging over uncertain
parent configurations and counting shared intermediates once. Its top-16
configuration panel assigns omitted mass neutral compatibility; conditioning on
both focal parents preserves legitimate backcrosses. Freeze this path evidence,
then replace the initial reciprocal term with a converged reciprocal
solve on the path-adjusted scores (maximum log-message change `1e-8`, at most
8192 undamped iterations, then a 4096-iteration damping-0.5 restart if needed;
the residual is measured before damping; failure of both attempts is an error in
the full-data fit, while a failed bootstrap/LOCO resample abstains and is counted). The final messages do not recompute the
supporting path factor. Explicit chronology
exempts that parental side without exempting an uncertain co-parent. All these
terms are recomputed during bootstrap and leave-one-chromosome-out fits.
Exposure requirements, explicit
eligibility, the fixed top-20 pair panel, fixed state priors and release
thresholds remain unchanged for chromosome-resampling release. A released
inferred edge additionally needs a decisive full-data orientation: the exact
reciprocal comparison of the pair, with all other converged messages fixed,
must give the reverse orientation at most 2% (3.9 nats); near-tied
orientations stay unreleased, and caller chronology is exempt.
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

## Founder allele refit after painting

Assembly never re-calls an allele: a chromosome row copies the alleles of the
block-local rows on its path. Where block-level model selection merged founders
that differ at a few sites, the row of the founder without its own local row
carries the other founder's alleles there, for all of its carriers. After
painting, the `allele_refit` stage performs one conditional M-step given the
fixed painting and the same normalized genotype likelihoods. At each site, a
row's allele is scored over the samples painted as carrying it (both tracks
named, positive depth, and not flagged by the sample-quality QC on that
contig, as in founder construction) with the painter's robust observation model,
`log((1 - eps) * P(reads | dosage) + eps / 3)` with `eps = 0.01`, holding each
sample's other painted allele at its current value. Each site is updated by
coordinate ascent (three sweeps) over single-row flips and joint two-row flips.
A move is accepted only if its gain is at least
`AlleleRefitConfig.minimum_gain_nats` (default 5 nats, a fixed prior against
change). The two-row moves correct allele swaps between two rows: a swap keeps
the genotype of every sample painted with that pair, so it produces phase
errors, and a single flip loses likelihood on those samples. On the simulations
below, the result equals an exact
search over all allele vectors at each site, except at a few dozen of 14 million
cells. Unknown alleles are never filled; completion fills are refit under the
same rule. Only both-named painted pairs are used: pooled tracks (identical
rows, `LOCAL_ALLELE_EQUIVALENCE` or whole-component pools) carry no
row-specific evidence and are skipped with their partner, as abstentions are. Row paths, founder counts and the painting are unchanged.

Each component keeps two matrices. Released alleles feed evaluation (the
`allele_refit` founder row) and export; inference alleles are the same values
restricted to the frozen assembly snapshot's called cells, and family phase
decodes its scaffold from them. The painter and pedigree evidence still use the
frozen snapshot, so painting keeps its no-feedback rule and pedigree results do
not depend on the refit. No repaint/refit iteration is run; in a reviewed
one-chromosome experiment, a repaint and second refit added little.

On the std_n80 and stress_n80 simulations (baseline products, 5 nats), founder
allele errors fell from 32,466 to 16,052 (16,460 fixed, 46 broken) and from 21,871
to 6,710 (15,207 fixed, 46 broken). In the family-phase scaffold, genotype errors
fell by 79% and 59%, and phase switch errors fell from 1,082 to 116 and from 2,937
to 88. Most remaining errors are filler rows of locally rare founders
whose carriers are painted on another row, so no carrier evidence reaches them.
On real data the refit follows the reads, including at artefactual sites
(paralogues, excess heterozygosity); no site-quality mask is applied here.

## Family refinement and recombination

Family refinement conditions on explicitly supported Tier-B parent edges.
Its scaffold takes a named track's alleles from the (refit) row and a pooled
track's alleles from its chunk members: their common allele where every member
is called and they agree, unknown otherwise. Final-phase paintings carry the
member sets, flipped with the labels.
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
restarts the polisher from the painting-scaffold phase, not from the preceding
polished path. Samples with fewer than two observed parents are held in that
(displayed) orientation during polishing; their released internal frame, which
recombination decodes, remains the family MAP. If phase remains unstable at 520
iterations, retain that attempt and restart from the scaffold with half
damping, at most twice (0.5, then 0.25, then 0.125 by default). Tolerance
scales with damping so the undamped residual criterion is unchanged. Successful ordinary solves are unaffected. Each attempt
has its own resume state; final summaries record attempted and final damping,
and total solver iterations. Set `FamilyRefinementConfig.phase_retry_count=0`
to disable retries. If every attempt remains unstable, no final phase is released.

Root frames. A pedigree root's scaffold frame (the track order of its painting)
is the frame of its displayed phase and of the polish of its two-parent
children. With one child, a frame change of the root and a crossover of that
child explain the data equally well, so a root with one child keeps its
scaffold frame inside each component (its family phase chain changes frame
only at component edges), and a single-boundary root gauge move needs at least
two children changing at the boundary (`minimum_gauge_support`). After the
release, frames of roots with at least three children are repaired once on the
final family state (`refinement/frames.py`). For a bin edge, each child's joint
selector chain (the family model's emissions) is re-decoded with the root's
homologues relabelled from the edge on. A child supports a frame change there
if relabelling at the edge (at a flagged edge: anywhere in the flagged stretch)
raises its max-path score by at least `root_frame_informative_nats` and its
flanks do not deny a change; it opposes if its homologue is the same on both
flanks of a location zone (+-`root_frame_location_bp` around the edge, or around
the flagged stretch) whose own evidence is left out, the flanks reaching
`root_frame_window_bp` from the edge. Short tracts inside the zone therefore do
not count, and children whose own change lies a little away from the edge are
neutral. Flagged edges are those where the root's own data do not carry the
orientation (painting ties, abstained or heterozygote-free bins); their
scaffold-error prior is 0. Where the root's family MAP frame agrees with its
scaffold over the window, both are relabelled from the edge to the end of the
component when at least three children support it and none opposes
(all-or-none, as a frame error moves every child), the prior plus the
children's score changes at the edge is positive, no other parent with at
least three children of its own is mixed in the two bins next to the edge (a
reference-order junction switches children binomially), the zone does not reach
a component end, and the edge lies outside supplied protected intervals
(breakpoint TSV `chrom, start, end, reason`, the recombination order-breakpoint
format; argument `protected_intervals` or `BHD_FAMILY_PHASE_PROTECTED_INTERVALS`;
rows whose reason starts with `reversed_segment` are not breakpoints). At a frame
change the family solve made against the scaffold, the same votes decide:
unanimous support for undoing it reverts the internal (decoder) frame, and
unanimous opposition copies it into the scaffold. Scaffold changes enter
before a final polish, so the displayed phase, the polished children and the
recombination decoder share one root frame there; frames are released in the
painting's coordinates, and the product lists every decision
(`root_frame_repairs`). A child with one observed parent releases its family
MAP frame, and where it is heterozygous at the parent's heterozygous sites
its meiosis observations follow that frame, which belief propagation aligned
with the root's (possibly wrong) homologue labels. So when a root's internal
frame changes, each such child's frame is changed with it inside every run of
these sites (bounded by sites where the child is homozygous, which are data)
that contains the change: the child's observations keep the family solve's
pattern relative to its homozygous sites on both sides; where the change
reverses the parity between those sides, the run's own switch nearest to it is
dropped, or, when the run has none, the switch at the change is kept, as the
data require one (`child_frames` records follow, cancel and keep). Children
with children of their own are recorded, not changed.
`FamilyRefinementConfig(root_frame_repair=False,
fix_single_child_root_frames=False, minimum_gauge_support=1)` restores the
earlier behaviour bit for bit. A real scaffold frame error of a single-child
root now becomes one false crossover call; the two cannot be told apart.

This is a phase point estimate, not a calibrated marginal posterior. Full family
probability tensors and a separate imputed source product are not produced.
The polisher scores copy consistency on called scaffold alleles only, so with
default settings the called phase does not depend on the family messages.
Family gap fills (0.98 context threshold) are not copy evidence: where a parent
has no reads they are imputed from its children's current phase and would
confirm that phase, including family-MAP switch errors. They never become
published allele calls. `PhasePolishConfig.polish_from_scaffold=False` restores
the earlier polish (gap fills as copy evidence, every sample from the family
MAP). Root
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
Candidate tracts are screened per meiosis from supported alternating origin
runs: a run between two opposite runs, or the run between an individual's run
boundary (chromosome end, component change, painting-coverage gap or oversized
marker gap) and its nearest supported switch. Tracts longer than
`maximum_candidate_bp` are not proposed; the same objective scores every one.

Maps report posterior expected crossover counts, called interval-censored
events and observable meiosis-bp exposure separately. Rates in weakly observed
regions can remain prior-sensitive; no exposure is not a measured zero rate.
Neither family inference nor the estimated map feeds back into earlier stages.

Final calls inside a long run without reads are imputed from the painting, not
observed. A sample's run of at least `RecombinationMapConfig.read_free_tract_sites`
(default 1000) consecutive zero-depth sites, taken from the observation source's
positive-read-depth mask, is therefore removed from that sample's coverage for
the shared-orientation fit and the per-meiosis decoder, like a painting gap:
its meioses, as parent or child, have no observation, exposure or called
crossover inside or across it. Shorter zero-depth runs remain observations; 0
disables the rule.

Reference-order junctions. The map assumes reference order is meiotic order.
Where a reference segment is misplaced or inverted, a meiosis changes
transmitted homologue at the junction whenever it has an odd number of true
crossovers between the two loci the reference joins: with probability
1/2(1 - e^-2d) for loci d Morgans apart, so about half of all meioses for
unlinked sides. These are real transmission changes in reference coordinates,
not crossovers at that position. Three outputs address them
(`recombination/junctions.py`):

- Reported call intervals. Painting and phase are rigid inside a painting bin,
  so a change inside a bin is placed at one of its edges. A call whose flanking
  supported markers straddle a bin edge is reported with the two bins around
  that edge (`reported_left_bp`, `reported_right_bp`); calls, counts and the map
  are unchanged.
- A truth-free junction detector that reports and never masks. Calls of at
  least three parents chained by overlapping reported intervals form a
  candidate locus. For each meiosis decidable across the locus, y is the parity
  of its calls there, modelled as y = s XOR f: s ~ Bernoulli(q) is a real change
  of transmitted homologue and f (prior 1/2, one per parent, shared by its
  children) a frame flip of that parent. Ordinary recombination, hotspots
  included, bounds q by a ceiling: a point hotspot (5%) combined with the
  background map parity across the locus. The mixing statistic is the profile
  log likelihood ratio of free q against q at most that ceiling, so all-or-none
  parents (frame errors) and hotspot-like single calls give no evidence. The two
  ends of a displaced or inverted segment switch in the same meioses; the pairing
  statistic compares one locus predicted from another (up to a discordance
  probability of 0.5-25%) with independent switching, takes the weaker
  direction and subtracts log(number of locus pairs on the chromosome). A locus
  is flagged when the larger statistic reaches ln 100. Likelihood ratios are
  conditional on the calls (no reads), assume independent meioses given the
  parents' flips, and are not calibrated posterior odds; power is limited by the
  number of informative meioses and by q itself (junctions joining loci a few
  cM apart look like hotspots).
- Optional order breakpoints (running guide, Reference-order breakpoints). At a
  supplied interval the sites of the overlapping painting bins leave the map axis
  and the component is split, so the two sides are treated as unlinked: no call,
  exposure or shared-orientation tract crosses, at the cost of those bins'
  exposure. A supplied breakpoint also hides any genuine structural polymorphism
  there, and decoding a segment on its own can expose junctions inside it. The
  upstream stages do not use the file; their panels, paintings and frames still
  assume reference order across it.
