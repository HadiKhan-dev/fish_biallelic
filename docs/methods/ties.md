# Ties, released ambiguity and stated priors

A tie is a decision whose alternatives the data do not separate. Either they score exactly the same (for example two
panels that differ only at alleles no sample observes), or they differ by less than the rounding noise of the compared
sums. The release policy handles ties by the following principles; the acceptance section below
distinguishes implemented treatments from remaining conventions and unfinished validation:

- **No decision that selects between different biological answers depends on:**
  - array, row or sample order;
  - hash order;
  - the thread count or the CPU instruction set;
  - an arbitrary content-key order.

  Content keys remain only as canonical labels of the same answer, for example the row order of a panel.
- **Alternatives that more data can separate are carried forward** as a bounded, order-invariant set, and a later
  comparison decides between them:
  - discovery's tied clusterings and panels;
  - local selection's tied starts, bank members and refits;
  - assembly's tied joins and refits.

  A bound on such a set is itself order-invariant. Discovery admits or omits whole classes;
  the bank admits or omits whole endpoint families and tied classes. Overflow is a computational
  limit, not biological ambiguity, and does not suppress an existing valid incumbent's calls.
- **Ambiguity the data cannot resolve is released as such**, not replaced by a conventional point answer:
  - founder alleles that no carrier observes are written as `.`;
  - painted switches inside stretches where the founders are identical become intervals;
  - a phase set breaks where family evidence does not decide the orientation;
  - parent slots follow recorded sexes, and are unordered otherwise;
  - tied alternatives left at the end of a stage are flagged and released with the decision taken by a stated prior.
- **Stated priors decide the remaining exact ties.** They are model preferences with a biological or statistical
  reason, listed in [Stated priors](#stated-priors). They are never index or content order.

Two acceptance tests check this on every release candidate:

- **The noise loop** reruns the whole pipeline under input perturbations that should not matter:
  - 1e-12, 1e-9 and (release 2.13) 1e-6 relative genotype-likelihood noise;
  - value-consistent noise;
  - permuted sample order, in memory and in the input file;
  - another thread count and another numerical environment.

  No biological outcome may change.
- **The convention reversal** reruns it with `HAPLOTYPES_TIE_REVERSE=1`. This reverses every remaining arbitrary
  convention: index order, content-key order, binary default states, sample-name order and keep-first. The stated
  priors and canonical labels are not reversed. Released outputs must equal the default run's, except inside released
  ambiguity. A residual difference marks a convention that still selects biology; the remaining ones are listed in
  [Remaining conventions](#remaining-conventions).

## Near-zero founder-count margins (2.7 decision 3)

Do not add a heuristic dead band. Existing numerical tie classes govern comparisons
between candidates: retain a complete family or report its unresolved alternatives.
Margins separated beyond the applicable quantum remain data. The `count_margin`
ranking quantum is 2^-26 in units of founder cost; the effective-carrier point rule
still tests both margins strictly above zero. Bootstrap ambiguity is a separate result,
not a new acceptance threshold. Changing that rule requires harm demonstrated against
simulation truth; cohort labels and assumed G0 parentage cannot supply that truth.

## Evaluating exported phase sets (2.3)

The `export_phase_sets_v2` evaluation follows the sample VCF's phase-set identity: a new set starts at a component boundary or an explicit per-sample `phase_breaks` boundary. A displayed-undetermined site is unphased and contributes its unordered allele error, but it does not split the phased flanks unless an explicit boundary is present. Each set gets one strand alignment, including phased sites on both sides of such an unphased stretch. Sets with at least two phased called heterozygotes contribute to the reported count and span N50.

Earlier evaluation omitted explicit boundaries and split at undetermined cells. Baseline and candidate comparisons must use the same corrected metric version. Component-aligned and union painting-tie metrics remain independent checks; releasing more PS breaks does not remove their orientation penalties. Release validation reports PS errors together with set count, N50 and callability.

## Family-phase release for 2.3

The stage6ab family-phase implementation retains the accepted release semantics by identity-tested kernel and scheduling changes. Whole-stage resource measurements and the incomplete final pipeline acceptance coverage are reported separately.
The release policy below records the approved behavior; it is not a claim that the
partially composed integration tree has passed the release gates.


Family phase derives its internal coordinates from pedigree depth and discrete scaffold/observed-read data, refining identical keys by pedigree neighbours. Sample names do not order the search. Both coordinate directions run both MAP schedules, giving an unordered set of four initial endpoints. Each receives one continuation before shared objective tie classes rank their final exact J. Initial ranking is diagnostic only. Each endpoint supplies its coordinates and schedule through continuation, repair, polish and evidence witnesses; its donor is the other schedule on the same axis. Final score-tied alternatives are aligned to the caller axis before projection.

Independent transplant clusters share no objective factor, including root marginals and supported-marker selector transitions across gaps. All gains use the same start. The total continuation bound is four; unvisited tied combinations are recorded as search incompleteness and do not themselves mask calls. Only evaluated retained releases add ambiguity masks, projected onto observable ordered calls (including half calls, excluding equal homozygotes and fully missing calls). This is not posterior enumeration. The reversal hook remains active.

Residual identical data/pedigree keys on nodes with no incoming or outgoing pedigree edges have independent unary/frame-chain factors. Their updates commute even when likelihood rows differ, so no likelihood ordering or equal-release requirement is needed. Other duplicate groups need an exchangeability certificate and identical member-specific releases. Cases that cannot meet this bounded contract require joint duplicate treatment and are reported explicitly; there is no fallback to sample-name order. The panel collision census and the full reversal/renaming matrix must accompany acceptance.

Every two-parent painting join requires a family-context witness comparing both relative flank orientations with equal refit effort. Ties use the shared tolerance; no confidence margin is added. Both flanks must have supported orientation. Unsupported joins become explicit display phase breaks. Each joined boundary records its relative and flank margins. Display-only breaks do not reset recombination selector chains or censor decoder calls.

The MAP phase view and genotype-preserving polish are distinct estimators. In C1's fallback policy, every component-local disagreement run longer than the existing 8,192-marker long-tract span, with a fully called heterozygote, contributes its internal endpoints as display phase breaks. Whole-component strand swaps of observable ordered calls add no internal break, including when raw switches lie at different positions inside homozygous or fully missing stretches. The unordered disagreement set does not change the point estimate, calls, frames, decoder or uncertainty masks. It does not claim that an orientation has been corrected. An additional orientation estimator is deferred to 2.4.

Export keeps phase-set continuity across displayed-undetermined calls. C1 therefore tests the flanks of each such internal stretch after retaining and projecting all release alternatives. The two-parent comparison leaves undetermined target genotypes unconstrained; one-parent flanks reuse their summed orientation evidence; roots reuse accepted repair votes. Every retained branch must support the released relative relation, otherwise an explicit break is added. Witnesses and counts are recorded in the existing diagnostics.

Assembly carries evaluated alternative families into release. Proven disagreement can
split founder continuity and withhold locally differing alleles. An unperformed
comparison preserves the incumbent and records verification status; a computational
limit alone never establishes biological ambiguity. Complete-shadow enumeration uses
2,048 combinations and the shared higher-context allowance is sixteen leaves. Proofs
must hold under every sample permutation of the original fixed bootstrap draws.

## Common machinery

`maths/ties.py` gives every stage one way to say "these values are tied".

- **Tie quanta.** Each kind of compared quantity has a power-of-two quantum. It sits far above the rounding of the
  compared sums and above the typical movement that relative genotype-likelihood noise of `maths.ties.GL_NOISE` = 1e-6
  causes in the compared contrasts. Releases 2.1 to 2.12 placed the quanta against 1e-9 noise, by a valley census of
  real and simulated runs; under a 1e-6 jitter, compared values left their tie classes and changed, for example, 144
  of 200 K30 discovery blocks and 19 to 37 discovery blocks per std or TroMau run. Release 2.13 (owner decision: no
  decision tolerance below the noise floor) re-examined the decisions of stages 1 to 5 against 1e-6 noise and
  raised those found below it to the smallest power of two at least 16 times their contrast's motion
  (`maths.ties.noise_quantum`, `NOISE_MARGIN = 16`). The motions are measured on frozen contrasts of 1e-6 jitter
  runs where available, otherwise the 2.1 census times 10^3 (a linear-response extrapolation, checked on the replays
  listed); a relative factor without a recorded motion is its 2.1 value times 2^10, validated by replays (the
  discovery factors). Typical matched-history motions and linear-response extrapolations guide the tolerances; bounded replays test the resulting policy. This is not a worst-case noise certificate. The factor 16 is a safety margin, not a significance level: differences smaller than a quantum
  are decided by content or a stated prior even where their sign is stable. The kinds and their quanta are:
  - block NLLs and cavity scores: at least 2^-10, and at least 2^-20 of the block's null NLL;
  - H-step allele costs: 2^-12;
  - "improved" tests: 2^-10;
  - per-sample sums: at least 2^-13, and at least 2^-26 of the scale;
  - per-cell evidence: 2^-14;
  - probabilities: 2^-15.

  Stages with their own scale use the same construction. Founder refinement's quantum is max(2^-9, 2^-26 |F|) and its
  acceptance threshold 2^-9 nats (refinement fit contrasts move by 9e-5 nats at the 99th percentile under a 1e-6
  jitter; a typical-motion census, not a certificate); local
  selection's decision tolerance stays 2^-10 (a search round's gain moves by about 5e-5 nats at the 99.9th percentile in the matched-history census); the painting
  Viterbi tie is 2^-15. The join quanta keep their 2.1 values: frozen join contrasts move 32 to 63 times less than
  them under a 1e-6 jitter, and no final join changed on identical inputs. Convergence tolerances of fits,
  float-resolution margins of exact bounds and stated statistical thresholds are not tie quanta and are unchanged.
  Not yet re-measured: founder refinement's guard tolerance (a scientific test on the guard's energy, 1e-6) and
  local selection's flip-acceptance tolerance (also its convergence tolerance, 1e-6); family phase's chain ties and
  gain quantum are outside the validated stages.
- **Tie classes.** Values are sorted exactly, and consecutive values within a quantum chain into one class.
  - **Width guard.** A class wider than four quanta is reported and ordered by value.
  - **Boundary guard.** A gap near a quantum is reported.

  The members of a class are the tied set. They are carried forward, decided by a prior, or flagged. They are never
  ranked by content, except for the explicitly bounded proposal conventions below
  (the residual edge cover and assembly beam cut) or a canonical label of an equal answer.
- **Threshold bands.** A value within a band of a threshold, for example a probability of exactly 1/2 or a
  contribution of exactly 0, is given the threshold's conservative side (a stated prior).
- **Reversal hook.** `maths.ties.reverse_conventions` reads `HAPLOTYPES_TIE_REVERSE`. The helpers `conv_key`,
  `conv_index`, `conv_order`, `conv_first`, `conv_argmin`, `conv_argmax`, `conv_state` and `conv_sorted` route every
  remaining convention through it. Without the variable they are the identity.
- **CPU independence.** Logarithms and exponentials in tie-relevant kernels use the scalar C library, and
  `NUMBA_DISABLE_INTEL_SVML` is set at import. No tie-relevant decision uses BLAS.

## Principled treatments by stage

| stage | convention addressed | treatment |
| --- | --- | --- |
| discovery ([discovery.md](discovery.md)) | HDBSCAN merges of equal-weight links ordered by a content key; complete linkage on noise | level-wise condensed-tree groups and a polynomial maximal edge-clique cover supply candidates; the cover retains every compatible pair before per-component size budgets, but not every larger group; D13 reverses greedy content order; the block score decides |
| discovery | NLL-tied panels decided by panel bytes | the tied panels are released with the block (`tied_panels`) and offered to local selection; archive beams keep whole NLL classes |
| discovery | founder alleles no carrier observes: the seed's allele | released unknown (`latent_unknown`) |
| discovery and candidate bank | tied A-step states assigned by row order or reduced to their meet | uniform fractional evidence over every tied state in the winning fewest-unknown-copy category; weighted H fields, cavity hold-out and release |
| discovery and candidate bank | founder updates choose one ordering of exact usage ties (D3) | complete endpoints of two consistent schedules, branching only on an encountered positive-usage tie; shared exact A work and certified identical H transitions, with schedule-specific fit certificates |
| discovery | dosage argmax chooses a single genotype (D8a) | low/high dosage vectors from the complete winning genotype class seed initial, direct and subtraction proposals; deduplication before unchanged class budgets |
| discovery | one population-half hardening chooses proposal starts (D8s) | both coherent hardenings are unioned and deduplicated before existing fitting and complete-class budgets; co-winning fits remain represented |
| discovery | residual chooses one tied dominant partner (D10p) | every eligible partner route receives the original observation once at its original weight; global observations are not multiplied |
| discovery and candidate bank | first mean-field start wins a training-ELBO tie (D12) | equal weight per eligible endpoint-equivalence class, with mean predictive likelihood within each class before the log; a deliberate change to the approximate predictive functional only in tie cases |
| local selection ([local_selection.md](local_selection.md)) | the path search starts from the last non-empty source's cavity-best panel | fit every source's cavity tie set, then search the best initial path-objective class, with fewer founders as the prior; compare completed outcomes by that objective |
| local selection | one nearest row or nonmember consensus selects exact-screened merges | every integer-distance minimizer is exact-screened, with the union deduplicated; unchanged likelihood, objective and refit caps |
| local selection | bank, refit cut and residual ties by content and order | the BIC tie set is kept whole; bank score caps admit complete overlapping endpoint families and tied NLL classes at unchanged caps; whole path-refit classes are retained (cap 16, overflow counted); residual ties give a consensus panel whose rows not shared by every alternative are withheld (`tied_alternatives`, `withheld_by_ties`) |
| assembly ([assembly.md](assembly.md)) | left-anchored batching | orientation-symmetric batching |
| assembly | join moves inside a log-likelihood tie class by path tuples or panel keys | the batch linker's path score decides before any label; painting-dependent proposals run under both traceback conventions and are merged |
| founder refinement ([founder_refinement.md](founder_refinement.md)) | tied refits decided by content | members that disagree on K take the smaller K (prior); otherwise the member whose margin most strongly supports the decision; remaining equal-F alternatives are flagged (`ambiguous_k`, `alternatives`) |
| founder refit ([founder_refit.md](founder_refit.md)) | the first acceptable move in row order | mutually exclusive tied moves release the affected cells unknown |
| painting ([painting.md](painting.md)) | a switch inside a stretch of identical founders placed at the leftmost edge, and the bins named by the prior | the switch is released as an interval (`SWITCH_INTERVAL`), and the other track is gated on the pair-consistent posterior; the per-sample segments are published |
| pedigree ([pedigree.md](pedigree.md)) | Parent1 is the lower sample name | parent slots follow recorded sexes (dam, sire); otherwise they are unordered (`ParentSlotBasis`) |
| family phase ([family_phase.md](family_phase.md)) | frames left free by the data took state 0 and the leftmost switch position | `frame_undetermined` marks frames another tied configuration sets differently, and the export breaks the phase set there |
| export ([usage](../usage.md#export)) | unobserved founder alleles written with the seed's value | written `.`, with counts in the manifest and the founder VCF header |

## Stated priors

These decide exact ties. They are not reversed by the reversal test.

| prior | where | reason |
| --- | --- | --- |
| fewer founders (smaller K) | discovery selection, the local-selection bank and round winner, founder refinement at an exact-zero count margin, count cycles | parsimony: at equal fit the data do not support the extra founder |
| fewer switches; fewer parents | the painting and family-phase Viterbi tie rule; the pedigree graph's replacement rows | at equal likelihood the recombination and transmission model already favours fewer events, because a switch probability is below 1/2 |
| keep the current state inside a tie class | search climbs, keep-incumbent tests, the family MAP's acceptance (a gain must exceed the quantum) | an ascent moves only on evidence, so the optimum is a fixed point: noise cannot make it move while the compared gain stays inside the band |
| the conservative side of an exact threshold | release, supporter, carrier and unknown-count bands; the abundant-route test | at exactly the threshold, withhold rather than call |
| abstain | pedigree winners, completion links and neighbours whose released alleles differ, export `.` at equal totals | no released answer where the data give none |
| more observed sites and more supporting samples first | discovery's data-birth ranking and residual emission | the better-supported candidate is tried first; candidate sets are kept whole up to their caps |
| fewest unknown copies first; fewest flips first | discovery's A-step band; the local-selection site posterior | the explanation that needs fewer unknown copies or changes is preferred at equal cost |
| pure-mosaic deaths first; flips before swaps | assembly count steps; founder refit | simpler explanations first |

Canonical row orders and panel keys used solely as identities, and the gauge of the two
tracks, are labels. The bounded proposal content orders are declared separately below.

## Status is not ambiguity

Discovery reports ranked truncation, computational incompleteness, unresolved initial
search, numerical non-convergence and local certification separately; their definitions
and effects are in [discovery status](discovery.md#interpreting-search-status). Only
evaluated conflicting alternatives establish biological uncertainty. An omission alone
cannot justify masking an incumbent. In particular, the 488/9,979 historical TroMau
blocks with mean-field failure somewhere in scoring do not establish 488 failed selected
panels. Scientific equivalence checks compare decisions, complete alternatives, calls
and missingness, while reporting continuous drift; floating-point bit equality is not a gate.

Selected-versus-other failure counts are now persisted; optional reversed search and
escalation are removed, while acceptance-only diagnostic hooks remain.
W3's D8s proposal union is accepted, and the bounded D9/order/solver findings are
recorded below. I6 broader observational-equivalence collapse remains analysis only;
G0 inclusion is retained.

## D8s family and bounded 2.5 closure

Population-half ties contribute both existing coherent hardenings as one complete family
for K1 cluster/dosage seeds, data births and soft-residual hardening. All tied sites move
together between those two members; the algorithm does not enumerate independent sitewise
completions. Union/dedup precedes existing fitting and complete-class admission. Objectives,
priors and budget values stay fixed; active paths/counts and calls can change. The accepted
gates have 4/4 inactive identities and 5/5 active hook-order identities. Low-coverage truth
differences are small and mixed, accepted without an accuracy-improvement claim; details
and empirical soft-hardening limits are in [discovery](discovery.md).

Bounded v2.5 discovery analysis closes D9 with 0/5 released changes on three K30 simulated and two TroMau blocks despite 40/81 changed cut sets (28/81 changed best cut weights); the three simulated blocks retain 16,256 calls, 6 called errors and 1,750 wrong-or-missing cells out of 18,000 truth cells. D9p/D3/central sample order has 0/15 released changes in the fresh controls. Across four retained mean-field cases, 5/53 scored panels fail only in losers, with zero selected or nearest-competitor failures; a stricter solve clears 470 failed held-out/site cells with maximum score drift 3.13e-10 and no class, K, panel, alternative or call change. I6 remains analysis only: the captured TroMau binding class has 50 omitted starts, 44 assignment-masked content classes and 50 distinct full-state signatures, so this capture supports no all-future-context quotient. These are bounded evidence closures, not universal invariance or full-pipeline proofs.

## Equivalent assembly seed witnesses (2.7)

An accepted seed endpoint is an equivalence class of all evaluated seed witnesses
with the same canonical settled panel. Their unresolved source families are retained
as a union; the diagnostic seed chosen to name the endpoint cannot choose which
ambiguity reaches release. This fixes the saved TroMau chr7 context-sidecar reversal
case. It is exact panel equivalence, not equivalence inferred from a tied likelihood,
and does not merge distinct endpoint continuations. See
[assembly](assembly.md#equivalent-seed-histories-27) for the scope and consumer limits.

## Scope of the 2.7 convention checks

The saved 2.6 census covers 7,748 unique joins in TroMau, std, K20, K30 and f3.
No recorded high-K seed group reaches the nine-seed cap; the largest has seven.
This does not establish that every main-coverer choice is unique. Refined-shadow
overflow is recorded in all five datasets: 223 calls, of which 181 remain unverified.
Opposite conventional shadows in eight bounded examples change sequence diagnostics,
but none changes the selected count outcome, selected ambiguity or incumbent panel.
The full shadow families are not thereby verified.

A per-rule refinement probe on f3 chr1 changes two path cells and 18 allele cells
at the initial K7 fixed point, at equal objective. Following that initial choice
through ordinary subsequent count/search rules returns the same K6 panel, four-panel
terminal ambiguity family, count margins, bootstrap bands and released alleles.
Branch histories and count-verification annotations differ. This isolates
one initial choice; it does not exhaust later tied trajectories or close global
refinement reversal. Unknown shadow verification stays unknown.

A fixed-input K30 context-refit reversal attributes the known posterior changes to
founder-allele conventions: maxima 0.00762084 at L1 and 0.08215804 at L2. These are
substantial persisted diagnostic differences. All 80 proposal calls, support values
and unknown-copy assignments remain unchanged. Local selection consumes the calls
and their founder paths, not the stored `posterior_alt` field; no changed downstream
input is demonstrated by that check. Retained template calls at low support must not
be interpreted as high-confidence posterior refits.

These checks distinguish exercised conventions, persisted diagnostics, retained
ambiguity and biological release. An unrecorded alternative is not counted as an
absent alternative. The seven join residues reported below remain a historical 2.2
count, not seven demonstrated current defects. The integrated release matrix is a
separate acceptance gate.

## Remaining conventions

The following counts are the **historical 2.2 baseline**, not validation results for the 2.3
symmetrized candidate. They identify the earlier errors that motivated this work:

| dataset | founder allele cells | genotypes | phase switches | meioses with other crossovers | Tier-B samples |
| --- | --- | --- | --- | --- | --- |
| TroMau, whole genome | 320,114 | 22,827 | 11,056 | 146 | 1 (status) |
| TroMau, three chromosomes | 3,225 | 1,361 | 1,713 | 0 | 2 (status) |
| std_n80 | 2 | 6 | 20 | 20 | 0 |
| stress_n80 | 0 | 0 | 13 | 26 | 0 |

### Discovery and bank status for 2.3

D1's compact meet is now a proposal summary only; weighted evidence uses the complete tied-state
set in every discovery and bank path. D3, D8a and D10p evaluate their complete documented candidate
families without permitting their former direction to choose a fitted answer. D3 may share an H
transition when tied update groups have no internal heterozygote membership dependency, or when the
whole sweep changes no allele. These certificates prove the identical next panel; they preserve the
usage-tie marker, both consistent trajectories and schedule-specific fit certificates. Fresh dense
A reuse and equivalent Boolean/integer proposal operations likewise reduce work without changing
the objectives, families or budgets. D12 uses the explicit predictive mixture described in
[discovery](discovery.md#cavity-score). After exact
full-vector deduplication, endpoints at maximum absolute coordinate distance at most the existing
mean-field convergence tolerance form connected equivalence classes. Classes receive equal weight;
within each class, distinct endpoints' raw predictive likelihoods are averaged. The existing
probability floor applies after the mixture, then the log. No founder-marginal average or held-out
choice of weights occurs. The unchanged eligibility rule prefers converged endpoints, uses all
starts only when none converged, and reports that failure. State priors, the NLL quantum, K prior
and set code are unchanged.

Exact endpoint counts were noise-sensitive in 1,456–2,021 of 16,000 sample-site solves in each of
12 K30 block/noise scopes. Counts of endpoint classes at the solver's existing 1e-10 tolerance
changed in zero solves across those scopes. This evidence required the pre-authorized convergence
scale for endpoint identity; no additional threshold is introduced. Full noise acceptance remains
separate from this identity check.

This bounded probe does not establish global invariance of endpoint multiplicity. Later parent7
K30 diagnostics under iid9 changed equivalence-class histograms in six of 200 blocks. Across
209,408,000 sample-site entries accumulated over scored panels, aggregate counts changed by
-50 in the one-class bin and +50 in the two-class bin. These are net histogram differences,
not fifty identified per-solve transitions. All 200 complete biological outcomes and selected
objective classes were unchanged. The existing equivalence scale is retained; no additional
threshold is introduced.

The diagnostic hooks remain named so interactions can be tested. Remaining proposal conventions
include D9 (maximum-cut and heuristic vertex order), D9p (proposal enumeration),
and central content/sample-order diagnostics. D8s now orders a complete two-member
family; both members are retained before admission. The bank's pre-fit start-backstop
still takes novel starts in enumeration order; its score cap now preserves complete families.
Other local-selection conventions are listed on its [methods page](local_selection.md#conventions-and-stated-priors).
D3's two schedules are not all possible schedules, and D8a alone does not establish general
REF/ALT relabeling invariance. No untested hook or interaction is declared harmless.

Candidate2 attribution assigns its full-reversal changes on K30, std and stress to D3, D8a,
D10p, D12 or their measured interactions, with zero unattributed residual. That attribution is not
a passing reversal result for the new combined implementation. A two-block K30 D3 canary has zero
complete biological changes under D3-only, KEEP=all, full reversal, noise, permutation and renaming;
an exercised two-block std D10p replay likewise removes its route-direction differences. These
bounded tests do not establish the whole-stage or whole-pipeline gates.

The source6 parent-ranking discovery matrix had zero biological differences across all nine
controls on fixed Tro220, K30, standard and stress inputs. Parent7's behavior-preserving speedups
were checked directly against source6 on reference, iid9 and value-consistent-noise outputs across
all four scopes: complete outcomes agreed, and reference selected objectives were identical.
These historical controls are separate from candidate8's own matrix and from full-pipeline
runs that regenerate evidence. The standard chr2:block511 difference is attributed to shared-binomial
calibration selecting different finite EM endpoints within its training-score band. On either fixed
likelihood input, parent7 and candidate8 give exactly the same discovery outcome under reversal.
The read-model candidate finishes all shared-binomial EM endpoints at stationarity under the
unchanged training likelihood and Newton policy. Equivalent endpoints require shared-logit
relative distance at most1e-13 and nuisance-logit distance at most the existing1e-8 stagnation
resolution. This equivalence is not an error bound. Maximum training likelihood, then the exact
lexicographic minimum full parameter vector, chooses the numerical representative. Content order
of training histograms fixes numerical sample reductions; nuisance rows are restored before use.
No held-out score chooses this representative. Distinct stationary models require explicit review and
cannot silently fall back to uncalibrated data. All 88 critical full-calibration profiles produce
bitwise-identical aligned likelihoods under start and sample order, duplicate and reversal controls.
The committed whole-run chr2:block511 outcomes both select K6 with identical complete panels and
objective, recovering one true founder and 200 correct calls relative to candidate8. Full standard
and stress reference/reversal discovery and all three local-selection rounds have zero biological
changes. Later phase differences remain; these stage results do not establish merged-pipeline
invariance.

At the provisional parent7 nominal 24/48 budgets, the symmetrized parent ranking had a K30
founder-stage regression. Own-score ranking improved K30 but worsened all seven stress founder
metrics relative to parent and is not adopted. All requested parent7 paired block-bootstrap
K30 and standard intervals include zero; this does not establish equivalence. Stress
missing-inclusive founder errors versus candidate2 have a positive 95% interval. The five K30
single-reversion ablations are complete: D8a and the narrow H/usage reference affect accuracy,
D3 trades coverage for called accuracy, and D10p/D12 contribute zero in that realization.
Full released calls, bank reversal, Tier-B and merged-pipeline accuracy/resource checks remain
separate. Discovery at 2x 2.2 and whole TroMau at 47:59 are timing guides; report actual ratios
and quality gains for any excess. The memory limit remains 62.55 GiB. Historical 2.2 residue
counts above are not acceptance results for the provisional 2.3 pipeline.

Local selection exact-screens the union of every integer-Hamming nearest-row and eligible
nearest-consensus minimum. Neither row order nor proposal order chooses a member. The refit
caps and likelihood/objective are unchanged; complete distance ties increase screening work.
The K30 m2:block0 reproducer exposed the former rule: the improving merge (12, 15) was
exact-screened only under reversal. The corrected common-input replay reaches the same K24
panel and local calls under full reversal, KEEP-all, iid12 seeds 1–3, iid9, value-consistent noise
and sample permutation. This block has no candidate-bank call or sample-name input. The same
three-round trajectory takes 32.192 to 33.100 seconds in two clean warm replays per version on
the same host with four numerical threads; the forward comparison includes an extra useful
round and takes 21.284 to 34.001 seconds. These are bounded block costs, not whole-stage costs.
Combined-candidate truth, stability and full-run resource gates remain separate.

### Search bounds and retained incumbents

Discovery's nominal class budgets admit a straddling class when the total remains at most twice
the budget; a larger class is omitted whole. Residual edge-cover groups are counted exactly
after lifting and deduplication, with a separate 64/128 size-class budget per connected
alternative component. No maximal-clique enumeration gate remains. Strictly worse classes
after the boundary are counted as ranked truncation and do not make the search incomplete. Replacement ranking and its
ordinary R = 48 / ceiling 96 and optional escalated R = 96 / ceiling 192 bounds are described in [discovery](discovery.md#replacement-admission); larger-budget
sensitivity must be reported for the adopted ranking.

A straddling tie-class or whole-gate omission sets `search_incomplete` and records cause, count,
scope and whether the count is exact or a lower bound. This states a computational limit of a
bounded search, not unresolved biology. Existing calls remain eligible for local selection. Only
a discovery block with no scored incumbent because all initial classes were omitted is
`unresolved`, and it supplies no calls. The bank keeps its separate unchanged score cap: overlapping
families and their tied score classes enter atomically, and an omitted first family leaves the
existing original latent incumbent in place. Complete competing fitted panels, when available,
still propagate as `tied_panels` with conservative raw calls.

## What a run records

Every stage that decides with tie classes stores `tie_events` in its records. For each decision kind (named
`<stage>.<decision>`) these count:

- `decisions`;
- `inside_tie` (the decision's class held several alternatives; for a kind ending in `band`, a value inside the band);
- `wide` and `boundary` (guard events).

The released ambiguity fields are:

- discovery: `ambiguous`, `tied_alternatives`, `score_margin`, `tied_panels`, `tied_unknown`, `latent_unknown`;
- local selection: `tied_alternatives`, `withheld_by_ties`;
- refinement: `ambiguous_k`, `alternatives` and `prior` per count decision;
- founder refit: `ambiguous_alleles` (cells released unknown);
- painting: `SWITCH_INTERVAL` segments, and switch-interval counts in the painting summary;
- pedigree: `ParentSlotBasis`;
- family phase: `frame_undetermined` (bit 0 decoder, bit 1 displayed frame);
- export: `unobserved_founder_alleles`.

Computational reporting is separate: discovery records `search_incomplete`, `unresolved` and
`omissions` with `kind`, cause, count, scope and `count_kind`; `ranked_truncation` does not set
incompleteness. The bank records `budget_bound`. Residual-route counters report original
observations, all route memberships and distinct pools without creating additional biological
support.

When the run completes:

- `ties/decisions.csv` and `ties/summary.json` sum the tie events per contig, stage and kind;
- the tie report aggregates discovery omission events and counts blocks with incomplete searches
  separately from unresolved blocks; counts of policy events are not unique omitted candidates;
- `painting/<contig>/sample_segments.tsv` lists every sample's released painting, intervals included;
- the founder VCF and the export manifest count the alleles written `.`.

`HAPLOTYPES_TIE_CENSUS` (a directory) switches on the diagnostic valley census that placed the quanta, and
`HAPLOTYPES_TIE_REVERSE_KEEP` keeps named discovery conventions at their default under the reversal
([configuration.md](../configuration.md)).

## Genome-wide pedigree admission (2.3)

For each child, retain every eligible parent within 60 existing information-temperature units of the best genome-wide raw M1 sum, including cutoff ties. Excluded identities receive the fixed chromosome profile whose sum is that genome cutoff; retained identities keep their raw scores. Apply this transform once, preserving every identity's multiplicity and contamination. Fix the profile before calibration, bootstrap and LOCO. Raw component and chromosome score checkpoints remain raw.

M2 includes every eligible unordered pair of retained identities, with no top-20/40 cap or partial tied class. The existing full eligible-pair denominator remains. This assumes individually plausible parents suffice; a weak M1 parent can complement another in a strong raw M2 trio. G4 diagnostics found such raw-likelihood cases, but no fitted or resampled G2 M2 winner used an excluded parent. This is not a universal false-negative guarantee. `parents_per_child` is legacy-only. Dense raw M1 and explicit floor rows remain in this reference; no near-linear claim follows from panel admission alone.

### Polynomial proposal conventions (2.8)

Residual discovery grows one maximal clique from every compatible edge, greedily in
hard-call byte order (REF, ALT, then missing at each successive site). This is convention
**D13**: `rev("d13")` reverses that order under `HAPLOTYPES_TIE_REVERSE=1`;
`HAPLOTYPES_TIE_REVERSE_KEEP=d13` holds it at default for attribution. Names and record
indices do not choose the order. Duplicate nonisolated rows are true twins, so their
relative order cannot affect membership. Reversal can change the proposed maximal
cliques and downstream births; it is a proposal approximation, not an equivalence claim.
The old `max_clique_enumeration` and `max_clique_calls` gates are removed. Exact counts
of deduplicated lifted cover groups drive the complete-class and straddling-class rule
within each connected component; a flooded component cannot suppress another's groups.
Where the former shared budget did not bind, changing its scope leaves groups unchanged.

Assembly's endpoint beam retains at most q paths, including within a score tie, by
canonical label-free content order under the central reversal hook. Constant Cartesian
families contribute only their first-covering members. These bounded proposal choices
may change the search pool even when the final fitted join is unchanged; they neither
enumerate the full tied family nor claim exhaustive ambiguity over it.

### Landmark proposal classes (2.3)

Discovery uses complete quantized farthest-anchor classes and complete nearest-anchor classes, without sample names or row-order representatives. Classes that exceed the fixed256 budget are omitted whole and reported as computational search limits. This alters proposal geometry and may alter latent panels; it is not biological ambiguity or an exact dense search. Final fitting and scoring still use all observations. Global residual pools can exceed256 records even when there are at most116 samples, so panel reuse requires measured inactivity or product identity rather than a sample-count assumption.

### Certified exclusion evidence (2.3)

Completed-window lower bounds can establish the existing genome-average Mendelian
exclusion decision; unproved cases finish exactly. This is an exact decision shortcut,
not a candidate truncation or new threshold. All observed-site counts remain exact.
Two diagnostic tables expose lower/upper endpoints and exactness flags, with NaN for an
uncomputed point value. Zero-target tables retain their original schema. See
[pedigree methods](pedigree.md#full-marker-mendelian-exclusion-of-m0-and-m1-counts).
