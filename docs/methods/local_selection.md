# Local selection

Local selection refits every informative discovery block under a normalised diploid copying model
and selects its founder panel by a drop / merge / add search on a regularised likelihood. It runs
three rounds per contig: an initial round from the raw [discovery](discovery.md) panels, then two
context rounds in which the current selection is assembled through assembly level 1 and through
levels 1 and 2, each assembled component's founders are refitted from every covered block's own
reads, and every block is selected again with those refitted founders as candidates. The panels of
the last round are the input of the final [assembly](assembly.md). These are local block
operations, not pedigree or family-phase feedback.

## Inputs and outputs

| | |
| --- | --- |
| input | the contig's `evidence` record in the founder view (samples flagged by the contamination QC are unobserved; see [evidence](evidence.md)), restricted to discovery sites; the discovery records (the latent panel of every informative block); the contig's genetic map; the local-selection configuration (including the copying redraw intensity); the assembly configuration (context assemblies) |
| product | one `LocalPanel` (`model/local_panels.py`) per informative block and round, in block order: positions, discovery sites, released calls `int8[K, L]`, ALT probabilities, favourable carrier support, and the per-sample unknown-copy summary that assembly's completion reads |
| records | `local_selection/<round>/...` for the rounds `initial`, `l1`, `l2`; `identity/local_selection/<round>` |

The stage has no options of its own; its constants are those of `LocalSelectionConfig`
(`local_selection/config.py`), listed in [configuration](../configuration.md). The genetic map
settings `[genetic_map] file` and `[genetic_map] rate_cm_per_mb` set the copying transitions, and
`[execution] batch_queue` lets helper allocations take a round's blocks (timing only).

## Rounds

| round | candidates of a block | context |
| --- | --- | --- |
| `initial` | the raw discovery latent panel; the search starts from it | none: every row is coded as a new haplotype |
| `l1` | the latent panel, the `initial` selection's calls and the proposals refitted from the selection assembled through level 1 | the assembled founders that cover the block |
| `l2` | the latent panel, the `initial` and `l1` selections' calls and the proposals refitted from the `l1` selection assembled through levels 1 and 2 | as above |

`LocalSelectionConfig.context_levels = (1, 2)` sets the assembly levels of the two context rounds.
Each round selects every informative block once; there is no further feedback after `l2`. The
same founder-view likelihoods and observation mask at the same discovery sites enter every round:
assembly contributes candidate sequences, never extra observations.

## Copying model

For a panel H of K binary founder rows over the block's L discovery sites, each of a sample's two
homologues copies a founder row or the *unknown* state (`local_selection/path_model.py`). The prior
state probability is pi = ((1 - u) f, u) with fitted founder frequencies f and the fixed unknown
mass u = `LocalSelectionConfig.model.unknown_mass = 0.01`. An unknown copy draws its alleles
independently as Bernoulli(1/2) at every site; it is not one shared unknown founder. At the first
site each homologue's state is drawn from pi; between adjacent discovery sites each homologue
redraws its state from pi with probability r = 1 - exp(-g dM), so the per-homologue transition is
T = (1 - r) I + r 1 pi^T, where dM is the genetic distance in Morgans between the two sites (from
the contig's genetic map, or the fallback rate without one) and
g = `LocalSelectionConfig.model.redraw_rate_per_morgan = 3.0` is the redraw intensity per Morgan.
The same intensity controls the block path HMM and the context-refit HMM. A redraw can select the
current state again, so g is not the observed founder-switch density: at stationary state
frequencies pi, the expected state-change intensity is g times (1 - sum_i pi_i^2). It is not
a generation label and is not fitted from painting counts. Assembly's independent
`AssemblyConfig.gap_horizon_scale = 3` limits linker and beam proposals; it does not set these
copying transitions. Both settings are internal constants, with independent roles, and neither
is a public TOML setting.

A bounded chr19 sensitivity study on std, F3 and TroMau retains copying intensity 3.0 and assembly
gap scale 3. Copying intensities 1 and 6 both worsen TroMau consistency against the G0 pseudo-truth;
F3 gains in genotype recovery at intensity 1 trade against worse founder recovery. The tested
assembly scales 1 and 6 offer no consistent gain. These painting-stage comparisons support keeping
the defaults, not treating 3 as optimal or interpreting pseudo-truth as known biological truth.

At an observed site, a pair of copy states emits the sample's normalised genotype likelihood of its
dosage: GL[H_a + H_b] for two founders, (GL[H_a] + GL[H_a + 1]) / 2 for a founder and an unknown
copy, and GL[0]/4 + GL[1]/2 + GL[2]/4 for two unknown copies. Unobserved cells emit 1.

The likelihood sums over all diploid copying paths (ordered state pairs) with their normalisation,
rather than choosing one fixed founder pair per sample. The two homologues share priors and
unphased emissions, so the forward messages stay symmetric: trajectories store only the upper
triangle while every ordered state's mass is counted. This symmetry is an exact property of the
model, not frequency tying or state pruning.

## Objective

A fitted (H, f) is scored by its log likelihood plus a coding regularizer
(`local_selection/objective.py`):

    J(H, f) = log P(GL | H, f) - log C(2^L, U) - c_claimed - log K - log(K + 1) + (1/K) sum_a log(K f_a)

- **Dictionary and context code.** log C(2^L, U) is the code length of U rows coded as new arbitrary
  binary haplotypes. In the initial round, and in blocks no assembled context covers, U = K. In a
  covered block of a context round, the refitted context founders may claim panel rows. The context
  supplies one local row per assembled founder of the covering component: founders whose local rows
  are equal remain separate claimers (release 2.12), and a founder with no called site in the block
  claims nothing. A claimed row is coded by reference to its founder plus its edits,
  log V(n_f, d) with V(n, d) = sum_{j<=d} C(n, j): the number of rows within Hamming distance d of the
  founder's n_f called sites (0 for an exact copy, n_f log 2 at most). The panel's code is its
  shortest such description (release 2.12):

      C = min over m of [ E_m + log C(2^L, K - m) ],

  with E_m the minimum summed edit code over matchings of m founders to m distinct rows (each founder
  and each row used at most once; any founder may claim any row). E_m comes for every m from one pass
  of successive shortest augmenting paths, O(min(F, K) (F + K)^2). Descriptions equal within a
  relative 1e-9 prefer more claims. Then c_claimed = E_m and U = K - m.
  - **Properties.** The code is a function of the row set: no row order enters. For fixed claimers
    and called masks, one changed call moves it by at most one edit-ball increment (at most
    log(n + 1) for a fully called founder of n sites), never by a whole free row. A row that
    complements a founder costs n_f log 2, not nothing.
  - **Scope.** The edit code covers only the founder's called sites: a claimed row's alleles at the
    founder's uncalled sites carry no code. This is a context-reuse penalty, not a complete prefix
    code, since neither the radius d nor the choice of claiming founders is coded.
  - **History.** Releases 2.1-2.11 let each founder claim only its nearest rows (a maximum-cardinality,
    then minimum-cost matching on those edges) at the shell code log C(n_f, d), and deduplicated equal
    local founder rows. 2.12 replaced both. A one-call change could move a claim between nearest rows,
    or merge two founders' rows, and so cost a whole free row (up to L log 2 nats).
- **K prior.** -log K - log(K + 1), the telescoping prior also used by discovery.
- **Frequency shrinkage.** (1/K) sum_a log(K f_a) is zero at uniform frequencies and negative
  otherwise; it amounts to one extra count spread uniformly over the founders.

This is a regularised best-panel objective, not marginal evidence for K or an assumed ancestral
founder count. Rows no founder claims (local recombinants, noise rows) pay the full dictionary code.

## Fixed-K fit

A fit (`local_selection/fit.py`) starts from a panel (uniform frequencies unless given) and
alternates generalised-EM updates, each computed from the forward-backward statistics of the
current (H, f):

1. a frequency step, f' = (c + 1/K) / (sum c + 1) with c the expected initial and redraw-destination
   copies of each founder, kept when the objective does not drop by more than the tolerance;
2. a flip step: the founder with the largest summed positive expected complete-emission flip gains
   flips all its sites with a gain above the tolerance; at a stationary point of these gains, the
   single exact conditional one-bit flip with the largest gain is taken instead.

The fit converges at a conditional fixed point: no frequency gain and no exact one-bit flip gain
above `LocalSelectionConfig.search.tolerance = 1e-6` nats. A flip that would make two rows equal
stops the fit, because changing K belongs to the outer search. Founders, or single exact flips,
whose gains lie within a relative 2^-10 of the largest (`local_selection.fit.FLIP_TIE = 0.0009765625`; 2^-20 placed
against 1e-9 GL noise before release 2.13, times `maths.ties.NOISE_SCALE` = 2^10 for 1e-6 noise) are tied. The data decide between them by the exact objective after each flip. If tied moves still
lead to different panels, the fit stops there (keep-current; termination `tied_flips` or
`tied_exact_flips`), so neither row order nor the last bits of the gains steer it. A fit is returned
with its rows in lexicographic order (frequencies and per-row posteriors permuted with them): the
likelihood and the objective do not depend on the row order, and two fits ending at the same
founders are the same panel to every comparison, tie-break and record downstream. The update backstop is
20 + 4K + 2L updates (`LocalSelectionConfig.search.update_backstop_base = 20`,
`LocalSelectionConfig.search.update_backstop_per_founder = 4`,
`LocalSelectionConfig.search.update_backstop_per_site = 2`), a reported limit. At fixed K the
dictionary term is constant, so a fit maximises the likelihood plus the frequency shrinkage; the
context code enters where fits of different panels are compared. A fit reports, per founder
allele, the exact one-bit conditional probability of ALT (every other bit and pi fixed, equal
prior odds) and the expected copies per site.

**Context-aware selection along the trajectory (release 2.12).** The GEM climbs the context-free
objective (likelihood plus frequency shrinkage), but a context round selects fits by the
context-coded objective, so a fit can pass through a panel the context code prefers and then leave
it. A context round's fit therefore keeps the best state under the selection objective along its
trajectory: the start and every accepted state. When that state beats the endpoint by more than the
decision tolerance, the fit returns it. Its frequencies are then refitted at its fixed panel by
frequency-only EM steps, each kept only when the selection objective rises; the context code of a
fixed panel does not depend on the frequencies. Such a fit is not a conditional fixed point
(converged False). Its record gives selected = trajectory_best, the update at which the state was
reached, and the selection objectives of the endpoint and of the selected state. Release 2.11
returned the endpoint only.

**Context snaps (release 2.12).** The GEM's exact one-bit steps use the likelihood only, so an allele the
data do not inform (an exact one-bit likelihood gain near zero) stays wherever the trajectory left it,
although the selection objective prefers it to agree with the claiming founder's context call: one edit
less saves log V(n_f, d) - log V(n_f, d - 1) nats of code. After the GEM, a context round's fit therefore
takes one-bit steps judged by the selection objective. One shortest description of the panel is chosen
in canonical order (founders by content and called mask, rows by content), so equal-cost choices do not
depend on row order. Each of its (founder, row) pairs proposes the founder's called sites where the row
differs from it. A proposal's lower bound is its exact one-bit likelihood gain (at fixed frequencies)
plus that pair's edit-code saving: the current description stays feasible after the flip, so the
selection value rises by at least the bound. A step is taken only when the largest bound exceeds the
decision tolerance; the proposals whose bounds lie within the tolerance of that largest bound form one tie
class (one band below the maximum, not a chain of near-equalities), decided by content (the resulting panel
with the smallest canonical bytes) and counted as `ls.snap.tie`. The chosen proposal's bound is then
strictly positive but may be below the tolerance, and the step is accepted only when the recomputed
selection value rises. Flips that would make two rows equal are not
proposed, since changing K belongs to the outer search. Each step solves one matching and scores one
panel, O(K^3 + K L), plus O(K L log K) per member of a tie class to find its canonical panel (only the
canonically smallest trial is kept). After each flip the frequencies are refitted at the fixed panel while the selection
objective rises. A fit records its snaps and tied steps; one with snaps is not a conditional fixed point
(converged False). This adds a selection-aware local improvement pass; it does not make the search complete
(a bit outside the matched pairs' disagreements, or one that would create a claim, is not proposed), and it
changes no likelihood, prior or code. A fit's `selected`, `selected_update` and `termination` describe the GEM
history before the snaps; `selected_selection` is the selection value of the returned panel.

## Panel search

The search (`local_selection/search.py`) starts by fitting the round's start panel. Every decision
between objectives in the search and between rounds uses anchored tie classes (`maths/ties.py`) of
the decision tolerance `LocalSelectionConfig.search.decision_tolerance = 0.0009765625` nats
(2^-10): values whose consecutive gaps are at most the tolerance are tied, and a tie is decided by
content (fewer founders, then the canonical panel bytes, then the order in which the candidates
were generated), never by the last bits of the objectives. The tolerance lies far above EM
convergence residuals and at least 16 times the typical motion of the compared contrasts under 1e-6
relative GL perturbations (a search round's gain moves by about 5e-5 nats at the 99.9th percentile
in the matched-history census of 1e-6 jitter runs, release 2.13; not a worst-case certificate), and far below the differences between distinct drop, merge and add outcomes. In a context round, when the start's fit
reaches the previous round's selected panel of the block (the *incumbent*), the search continues
from the incumbent's fit; when the incumbent scores better than the start's fit by more than the
tolerance, the search explores from the incumbent instead (refitted only when that improves it by
more than the tolerance), so it does not climb back to it. Each search round then:

- proposes every **drop** (one row deleted, frequencies renormalised), every **merge** of rows
  i < j (row j deleted, row i set to the occupancy-weighted consensus of the two rows' allele
  probabilities, its frequency f_i + f_j; where that probability is 1/2 within
  `local_selection.search.CONSENSUS_TIE = 9.5367431640625e-07` (2^-20 since release 2.13, 16 times
  the motion under 1e-6 GL noise; 2^-31 before, placed against 1e-9; the band now also covers consensus
  values that were distinguishable from 1/2 at 1e-9), row i's allele is
  kept) and every **add** of
  a candidate-bank row not in the panel (frequency 1/(K + 1), the others scaled by K/(K + 1));
  starts with the same panel and frequencies equal up to rounding are proposed once;
- screens drops and adds by their exact forward log likelihood plus the coded regularizer, and
  merges by an EM lower bound: the merged panel is the drop panel of either row with the other row
  replaced by the consensus, so the drop panel's posterior gives LL(drop) plus the replaced row's
  complete-data flip gains at the changed sites plus the expected-copy frequency term, and the
  larger of the two bounds ranks the merge. For every row k, the merges of k with its nearest
  rows and the merges not involving k whose consensus is nearest row k are double-removal
  candidates: the fixed-posterior bound credits no re-routing of carriers and can rank them far
  below their exact scores. From each of these two nearest-distance classes (Hamming distances are
  integers, so classes are often large: on K30 a merge's consensus usually equals one of its two
  rows) the member with the largest EM bound is scored exactly (release 2.17; tie classes of the
  decision tolerance, then content order; a choice inside a tie class is counted as
  `ls.rank.merge_representative`), so at most 2K merges per round are scored exactly and the others
  keep their bounds. Tied bounds do not imply tied exact scores, so the choice is deterministic for
  a fixed tie convention but not invariant to it. Until 2.16 every member of both classes was scored
  exactly, up to K(K - 1)/2 forwards (a K30 census found 76-277 of about 400 per screen).
  Likelihoods, objective, refit caps and stopping rules are unchanged;
- refits whole tie classes of each kind's screened scores, in order, until at least
  `LocalSelectionConfig.search.refits_per_kind = 8` proposals; a class that would take the kind past
  `LocalSelectionConfig.search.refits_per_kind_cap = 16` is left out whole and counted as an overflow (the block's
  summary records `cut_overflow`), so no member of a tie class is preferred to another. Each is refitted to a
  conditional fixed point, and the search
  moves to the best refit (the best tie class of the objective; inside it fewer founders, then the
  panel bytes, then the refit order) when it improves the selection objective by more than the
  decision tolerance.

The search stops when no refit improves; the round backstop 10 + 2K
(`LocalSelectionConfig.search.round_backstop_base = 10`,
`LocalSelectionConfig.search.round_backstop_per_founder = 2`) is a reported limit. After the
search, the incumbent is kept unless the search result beats its selection objective, both under
the current round's context code, by more than the decision tolerance (hysteresis): a context
round replaces a block's panel only for a real improvement, so a change elsewhere in the contig
does not move a block between equally good optima. A block whose search inputs (start, candidate
bank, likelihoods, map distances, settings and context founders) equal the incumbent's reuses the
incumbent's search. Fits, scores and merge bounds are deterministic functions of their
inputs; they are memoised within a block and carried to the block's next round. A context round's
fits depend on the context through the trajectory selection above, so their memo keys include the
context founders.

## Candidate bank

In the initial round the start is the raw latent panel and the bank its rows. In a context round,
the block's candidate panels are first completed: a partial row (-1 cells) takes the alleles of its
nearest latent rows on its called sites (every tied completion enters the bank; called alleles are
never overwritten). When equally near latent rows give different completions, the panel becomes one
source per combination of its tied rows' completions, so no donor is preferred. A panel with more
than `LocalSelectionConfig.bank.completion_variant_cap = 8` combinations keeps only its rows with a
unique completion as its source; this is counted as an overflow. Every source (the latent panel, each earlier selection, the newest proposals)
is refitted with discovery's fixed-K model and its endpoints are cavity-scored
(`local_selection/bank.py`). Every synchronized and refitted endpoint uses discovery's fractional
A step: each sample contributes uniformly over all tied states in its winning unknown-copy
category. Those same memberships feed the founder field, leave-one-out cavity scoring and the
raw-discovery release of each start. The compact meet pair is only a summary and does not discard
a tied sample's founder evidence. Samples without observed sites are removed from the cavity
likelihoods and their memberships together. Every bank path evaluates both consistent usage-tie
schedules when a positive-usage tie occurs, sharing exact A work and keeping every distinct endpoint
and lower-K collapse descendant in its source family. It also shares an identical H transition when
tied updates provably commute or the sweep changes no allele; both trajectories and their separate
fit certificates remain required (see [discovery](discovery.md#implementation-and-cost)).
The fractional allocation and schedule
families leave the BIC, switching-Viterbi NLL and subsequent path objective unchanged. The separate
[D12 cavity policy](discovery.md#cavity-score) deliberately changes tied-endpoint predictions to a
mixture with equal endpoint-class weights and mean predictive likelihood within each class before
the log; source cavity comparisons use that same functional.

The union bank and the drop, add and replace neighbourhoods of its best panels are then searched
under discovery's switch-penalised binned-Viterbi NLL in a BIC form,
K x `LocalSelectionConfig.bank.complexity_scale = 0.5` x (L / 200) x N_active + 2 NLL (lower is
better; N_active the samples with depth), until it improves by at most the block's `improvement`
tie quantum (2^-10 nats since release 2.13; 2^-14 before). The block's discovery tie quanta define the bank's tie sets, which are
kept whole (no member is preferred by its key). The selected panels are the best tie class of the
BIC score (kind `nll`), inside it the best class of the NLL, then fewer founders (a class wider
than the width guard keeps only the panels exactly at its best value). Every member's
neighbourhood is searched and every member's rows join the bank. The candidate starts are each
source's cavity-best tie set, chosen the same way from its cavity scores.

Score-cap admission preserves complete source families. Families sharing any endpoint are joined
into atomic components and ranked by their best actual endpoint NLL using the existing `nll`
quantum; complete tied classes are admitted together. The cap counts unique new endpoints. If the
next complete class cannot fit, it and all lower-ranked classes are omitted, `budget_bound` is
reported, and already valid incumbents remain. There is no doubled-cap allowance in this bank
policy. If no initial family fits, the existing original latent incumbent and its rows remain;
no BIC winner or new calls are invented. The separate pre-fit start backstop still limits novel
starts in enumeration order and remains a convention to test; complete endpoints of an admitted
start are never cut to an index prefix. Reported backstops, with K the
current panel's founder count: at most 512 + 64K novel starts per call
(`LocalSelectionConfig.bank.start_backstop_base = 512`,
`LocalSelectionConfig.bank.start_backstop_per_founder = 64`), 1024 + 128K admitted panels
(`LocalSelectionConfig.bank.score_backstop_base = 1024`,
`LocalSelectionConfig.bank.score_backstop_per_founder = 128`) and 8 + ceil(K/2) rounds
(`LocalSelectionConfig.bank.round_backstop_base = 8`).

The candidate starts are the cavity-best tie sets of every source, each member released with the
raw discovery rule and the block's tie quanta (released alleles, latent alleles where withheld). An
incomplete discovery search with an incumbent follows this same path. The exceptional block with
no incumbent because every initial class was omitted contributes no local panel; its unresolved
status and omissions remain recorded in discovery, and no calls are supplied to assembly. Each
candidate start is fitted. The best tie class of their selection objectives (fewer founders inside
it) is searched from every distinct fitted panel. A lower-scoring initial fit is not searched merely
because it belongs to a different source; a better initial objective does not guarantee a better
completed search. The incumbent can still supply the starting fit under the score/reached-panel
rules above. Completed outcomes are compared by the selection objective's tie classes, fewer
founders inside the best. Distinct panels still tied (`tied_alternatives`)
are released as their consensus. A founder row that every alternative contains keeps the calls all
alternatives' releases agree on, their mean ALT probability and their smallest support. The other
rows are withheld (`withheld_by_ties`). Unknown slots are the largest over the alternatives. The
next round's incumbent is the alternative with the smallest canonical key, a convention the released
panel does not depend on. In the initial round, discovery's NLL-tied alternative panels of the block
(`DiscoveredBlock.tied_panels`) join the latent panel as candidate starts, and in context rounds as
sources. The rows the add move may use are the raw latent rows, the candidate starts,
every member of the BIC tie set and the incumbent's rows. Neither the BIC nor the cavity score
replaces the path objective; they only supply starts and rows. Completed latent alleles are not
released as calls by themselves.

## Context rounds

**Context assembly.** Round `l1` assembles the `initial` panels through level 1 and round `l2`
assembles the `l1` panels through levels 1 and 2, with assembly's role CONTEXT: the same assembly
and completion as the final assembly, without founder refinement (see [assembly](assembly.md)).

**Coverage.** A context component covers a block when its sites are exactly the sites of whole
consecutive blocks. Every covering component is refitted, whatever its founder count; a component
that covers blocks only partially is skipped and its blocks keep their current selection as their
proposal, with every row coded free. A completed assembly round therefore does not imply that
every block receives a new proposal.

**Refit** (`local_selection/context_refit.py`). The component's distinct pre-fill rows are its
founders (-1 = uncalled), taken in lexicographic order, so the refit does not depend on the
component's founder order. Per sample, a bin-level diploid HMM has one bin per covered block and as
states the ordered pairs of the component's K founders and a background state, whose alleles are
Bernoulli(site background frequency) per copy, the background frequency at a site being
(1 + called ALT) / (2 + called founders). Transitions redraw toward pi = ((1 - u)/K, ..., u) at
probability 1 - exp(-g D) over the distance D between block centres (Morgans with a genetic map file,
else bp times the fallback rate), with u the unknown mass above and
g = `LocalSelectionConfig.model.redraw_rate_per_morgan = 3.0`, the same redraw intensity used
by the block path HMM. A pair's bin emission sums, over the bin's observed discovery sites at which some
founder is called, the logarithm of a robust genotype likelihood (1 - m) GL(pair) + m/3, with
m = `LocalSelectionConfig.context_refit.emission_mix = 0.01`.
The *cavity* weights of block b, its pair posterior from a forward-backward pass that excludes
block b's own emission, weight each sample in the shared founder-allele site posterior of the
assembly library, which refits every founder allele at each site of block b from that block's own
reads.

The site posterior is exact for at most
`assembly.founder_alleles.site_posterior.MAX_EXACT_BITS = 10` founders (all 2^K configurations
enumerated). Above that it uses the polynomial uncertain core: a posterior mode by loopy belief
propagation and single/pair-flip ascent from two starts; founders whose single flip and every pair
flip cost more than `assembly.founder_alleles.site_posterior.SITE_CERTAIN_NATS = 20.0` nats stay at
the mode; the remaining core is enumerated exactly when it has at most ten founders, otherwise over
a Hamming ball of at most `assembly.founder_alleles.site_posterior.SITE_BALL_CONFIGURATIONS = 1024`
configurations (radius at least `assembly.founder_alleles.site_posterior.SITE_BALL_MINIMUM_RADIUS = 3`).
There is no founder-count cap. An allele is released at posterior probability
`LocalSelectionConfig.context_refit.posterior_threshold = 0.99` with carrier support of at least
`LocalSelectionConfig.context_refit.min_support = 2.0`, by count (samples more likely than not to
carry the founder) or by carrier mass; nothing is released at a site whose two ascent starts end in
different comparable optima (multimodal abstention), or where the ball's shell guard withholds it.
The refit's founder prior is uniform, so a carrier probability can be exactly 1/2 by symmetry and a
carrier mass exactly at the threshold (a sum of halves): within
`assembly.founder_alleles.site_posterior.SITE_HALF_BAND = 9.5367431640625e-07` of 1/2 (2^-20 since release 2.13,
at least 16 times the motion of 1e-6 GL noise and below the first genuine values; 2^-26 before) a sample is
not counted, and a mass within that band of the threshold does not release (a fixed conservative
rule at a mathematical tie). An ascent flip, a ball recentring and the complement start's optimum
must raise the log posterior by more than
`assembly.founder_alleles.site_posterior.SITE_IMPROVEMENT = 0.0009765625` nats (the `nll` floor, 2^-10 since
release 2.13). The [assembly](assembly.md) page describes this posterior in
full. A withheld allele keeps the component's published (completion-filled) call when it has one.

**Proposals.** A covered block's proposal is the set of distinct local rows of the refitted
founders, in lexicographic order: its candidate panel. The refit and the candidate panel use these
distinct rows. The dictionary code instead has one claiming slot per assembled founder of the
component (release 2.12): each proposal row is recorded with its multiplicity, the number of the
component's founders that carry it (counted through both the component's and the block's merging of
equal rows), and a slot with no called site in the block cannot claim. The refit summary's
`max_founders` counts distinct refitted rows, not claiming slots. Proposals are recorded per round.

## Release rule

The selected panel's alleles are released from one more E-step over per-sample posteriors
(`local_selection/release.py`). For founder k at site l:

- the one-bit conditional probability of the panel's allele is 1 / (1 + exp(G_kl)), where G_kl is
  the log marginal likelihood ratio of flipping that allele with every other bit and pi fixed
  (summed over samples, equal prior odds);
- the *favourable* samples are those whose own exact flip gain is below
  -`LocalSelectionConfig.release.favourable_margin = 6.103515625e-05` nats (the `cell` tie
  quantum, 2^-14 since release 2.13), i.e. whose evidence favours the allele; a symmetric-evidence cell (one REF and one
  ALT read: GL(0/0) = GL(1/1)) gives a homozygous carrier a gain of exactly zero, inside this band,
  so it is never favourable; the carrier probability is the posterior probability that a homologue
  copies founder k at the site (a homozygous state counted once);
- the allele is released when its probability reaches
  `LocalSelectionConfig.release.min_call_probability = 0.85` and the favourable samples include
  `LocalSelectionConfig.release.min_supporters = 2.0` carriers, by count (samples with carrier
  probability above 1/2) or by expected carrier mass. The mass alone sits just below two for a
  founder with exactly two near-certain carriers; the count alone would withhold diffusely
  supported sites; so either suffices;
- **context witness (release 2.12).** The probability test is never relaxed, but in a feedback round an allele that
  passes it and fails the support test is released when the row is claimed by an assembled context founder in the
  panel's shortest description (the matching of the objective's dictionary code), the allele equals that founder's
  call, and the context refit itself released that call for at least one founder carrying this local row (posterior
  `posterior_threshold` with `min_support` carriers; not a call retained from its input or a completion fill;
  `ContextProposal.witnessed`). The block posterior splits the carriers of near-identical rows by frequency, so a row
  the context claims kept fewer than `min_supporters` carriers of its own and was withheld wholesale although its
  alleles are shared with carriers the refit identifies; context snaps made such rows common. The refit identifies
  carriers by linkage (cavity weights from the flanking blocks) but reads the same block's reads, and its founder rows
  were themselves learned from the data: this is conditional reuse of the evidence with a better carrier assignment,
  not independent confirmation, and the two thresholds do not combine into a probability of correctness. A
  support rule pooling the carriers of mutual nearest rows was tested and left out: on simulations it released the
  common row's allele at sites private to the founder the row matches (std, one-to-one matching of rows to true
  founders: 60 of 67 such rescues wrong), although its errors against the nearest true row were few.

Released alleles are the panel's own; others are -1, as are all sites that are not discovery
sites. Because the release depends on the round's context witnesses, an incumbent's release is reused
only when the selected panel and frequencies are unchanged, the round has no context witnesses and the incumbent's
release used none. The probability is a conditional one-bit quantity, not a full allele posterior or a
calibrated probability of correctness. The `LocalPanel` stores the fit's ALT probabilities and the
favourable carrier counts.

**Unknown-copy summary.** For each sample with depth, the largest MAP number (0, 1 or 2) of copies
drawn from the unknown state over its observed discovery sites (2 for a sample without depth), and
the block's `unknown_mass`, those copies over the 2 copies of every sample with depth. A site's
MAP count must win by more than `LocalSelectionConfig.release.unknown_count_band = 3.0517578125e-05`
(2^-15, the `probability` tie quantum since release 2.13) times the largest count mass, else the smaller count is
taken: completion's veto is all-or-nothing, and count masses can tie exactly. Assembly's
completion reads them as a spatial MAP summary; it is not a probability of any unknown state and
does not invent a constant founder pair for a sample.

## Conventions and stated priors

Where tied alternatives remain after every data comparison, the rules below decide. The *conventions* carry no
biological meaning and are reversed when `HAPLOTYPES_TIE_REVERSE=1`
(`maths.ties.reverse_conventions`; the release 2.2 acceptance test). A released result that changes under the
reversal was chosen by a convention. The conventions are:

- the novel starts kept when the bank's start backstop binds (the first in enumeration order);
- the fit kept per bank key (the first admitted);
- the internal representative of distinct panels left tied after the multi-start search (the smallest canonical
  key; the released panel is their consensus);
- the content order inside tie classes (canonical keys, panel bytes, generation and trial order; reversed centrally
  in `maths.ties.tie_classes`);
- the merge consensus allele at a probability of exactly 1/2 (row i's, the lexicographically smaller row);
- in the site posterior: the founder order of tied ascent moves and of tied ball recentrings, and the REF start of
  an exactly zero BP marginal.

In the context refit, duplicate founder rows share the consensus of their released templates (-1 where called
templates differ). Duplicate local rows carry their summed support and support-weighted ALT posterior. A bin's background-copy
count is the smallest count within a relative 2^-15 of the largest state-pair weight
(`local_selection.context_refit.BACKGROUND_COUNT_BAND = 3.0517578125e-05`; the `probability` tie quantum since release
2.13). These are data decisions, not
conventions.

The *stated priors* are model preferences and are not reversed:

- fewer founders inside a tie class;
- keep-current inside a tie class (the climb stops unless a gain exceeds the decision tolerance; the previous round's
  panel is kept unless beaten by more than it; a start that reaches the incumbent's panel continues from the
  incumbent);
- the conservative side of an exact threshold (the favourable band, the smaller unknown-copy count, the half bands
  of the context refit and of the site posterior);
- fewest flips first in the site posterior's ascent and ball;
- a GEM fit stops where tied moves lead to different panels (keep-current);
- the BP basin over its complement at equal optima, which are flagged multimodal and not released.

Proposal starts whose frequencies agree to within `local_selection.search.PROPOSAL_FREQUENCY_TIE = 1e-12` are one
start, not alternatives. Refits that reach the winning panel from different proposals are one answer, differing
only by EM residuals; the refit of the proposal with the smallest (kind, proposal panel) label represents it.
Canonical row orders and these labels are not conventions.

### Observed effects of the start cap and inner winner order

A census of 318,048 block-rounds across eleven datasets finds nine start-cap events in six
block-rounds, all on TroMau or K30. The cap is the 512 + 64K novel starts per bank call stated above.
There are 73 same-K distinct-winner ties among 311,074 inner winner decisions; six advance the
search and 67 are terminal. Both conventions therefore bind internally.

Replaying 223 flagged block-rounds with the global convention reversal and with reversal isolated
to the inner winner's content key changes no released local calls, K, missingness, favourable
support, unknown-copy summaries or local alternative/incumbent decisions. Native propagation
checks on K30 m1 (40 blocks) and TroMau chr7 (834 blocks) retain these results through all three
rounds. The start-cap check uses the global hook, not an isolated start-cap-only intervention.

K30 retains one changed latent allele that is uncalled in both releases, with maximum ALT-probability
drift 6.7e-6. TroMau's initial drift disappears after the first context round. These are measured
search-trajectory effects, not a claim that all numerical fields are identical. The evidence
does not justify a remedy for these two conventions and does not establish invariance on untested data.

The global hook also reverses assembly conventions. It changes one TroMau L2 context-assembly
`release_alternatives` record and some K30 context-proposal probabilities; neither effect occurs
under winner-only reversal, and neither changes the released local calls or decisions in these
checks. The [assembly follow-up](assembly.md#equivalent-seed-histories-27) is separate
from the two local-selection conventions. Final assembly, painting, pedigree, phase and exported
outputs are outside the propagation check.

## Records

| key (under `local_selection/<round>/`) | content |
| --- | --- |
| `blocks/<n:06d>` | chunk n, in write order, of up to 64 completed blocks: their `LocalPanel`s, fits and search outcomes, releases and summaries |
| `memos/<n:06d>` | rounds `initial` and `l1`: the same blocks' likelihood memos for the next round (written before the chunk) |
| `proposals` | rounds `l1` and `l2`: the context proposals and the refit summary |
| `selected` | round completion |
| `assembly/...` | rounds `l1` and `l2`: the context assembly's own records (joined batches are shared by all assemblies of a contig) |

The identity of the `initial` round binds the local-selection and discovery configurations, the
copying redraw intensity, the genetic map and the evidence and discovery record identities, with the
code of local selection and discovery only, so assembly changes leave the initial fits valid. The
context rounds add the assembly configuration and the previous round's identity. Readers:
`local_selection.records.load_panels` (the final assembly reads round `l2`), `load_selections`,
`panels_identity`, `round_complete`, `complete`, `progress` and `founder_panels` (per-round calls,
for evaluation).

## Implementation and cost

| module | role |
| --- | --- |
| `local_selection/stage.py`, `local_selection/rounds.py` | the three rounds; per round one pool task per block, or cross-node batches |
| `local_selection/path_model.py` | forward-backward of the copying model: lane kernels for scores and fit statistics, the per-sample release E-step |
| `local_selection/fit.py`, `local_selection/objective.py`, `local_selection/search.py` | GEM fit, objective, drop / merge / add search and memos |
| `local_selection/bank.py` | candidate bank (discovery's fixed-K fit and cavity score) |
| `local_selection/context_refit.py` | context refit on `assembly/founder_alleles/site_posterior.py` |
| `local_selection/release.py`, `local_selection/records.py` | release, `LocalPanel`, records |

Blocks are independent within a round. A round runs one task per block in a process pool over
shared evidence arrays (or as cross-node batches of consecutive blocks), and the context refit runs
one task per component. The path kernels process samples in lanes of up to 64 side by side. For
blocks with more than 64 samples, lanes narrow to 32 or 16 when all resulting lane blocks fit the
held thread budget. Each kernel caps its active threads at the useful lane count and restores the
caller's thread mask afterwards, retaining the CPU claim throughout. Uncached evaluations refresh
the layout when the budget changes. Sample results are combined in the existing order; released
scientific outputs and decisions are unchanged, with continuous drift reported in the
[performance checks](../performance.md#local-selection-measurements-for-26).

Cost in the founder count K, samples N and discovery sites L of one block (S = K + 1 copy states):

- **One forward or backward pass** is O(N L K^2): the redraw transition is separable, so each site
  costs O(S^2) per sample. A backward pass computes only the statistics its caller needs: expected
  copies, then complete-emission flip gains and occupancy, then the exact one-bit gains, which are
  computed only at stationary points and for the reported probabilities.
- **A GEM update** is a few passes, O(N L K^2); a fit takes at most 20 + 4K + 2L updates.
- **A search round** has an O(N L K^3) likelihood screen: drop forwards, add forwards,
  drop-statistic passes, merge lower bounds and at most 2K exact merge forwards (one representative
  per nearest-distance class, release 2.17). In context rounds each of the O(K^2) proposals also
  has its context-coded regulariser, a founder matching of O(min(F, K) (F + K)^2) for F context
  founders, which is not bounded by this figure. Refit admission still uses whole score classes,
  with target 8 and cap 16 per proposal kind.
- **The candidate bank** fits at most 512 + 64K starts per call with discovery's fixed-K model,
  completing up to two usage schedules per start only where a tie is encountered, with shared A
  work and certified identical H transitions (see [discovery](discovery.md#implementation-and-cost)).
- **The release** is one per-sample E-step, O(N L K^2).
- **A context refit** of a component with B blocks and L_c sites costs O(N L_c K^2) for its bin
  emissions and O(N B K^2) for the bin forward-backward. The site posterior costs O(N 2^K) per site
  up to the exact switch point K = 10 and polynomial work above it (O(N K^2) per belief-propagation
  iteration and ascent step, O(N (m^2 + 2^10)) for an exact core of m founders, O(N m^3) for a
  radius-3 ball); nothing in it is exponential in K above ten.

The numbers of search rounds, GEM updates and bank rounds are set by convergence; their reported
backstops grow linearly in K (and in L for GEM updates), so the worst case the backstops allow is
O(N L K^3 (K + L)) per block for the search. Memory per block search is dominated by the trajectory
scratch, at most 64 L (S(S + 1)/2 + S) float64 values per active thread (one lane block each), shared
by the block's sequential fits. The memory estimate retains width 64; layout transitions can briefly
retain old arrays. A context refit holds one (N, B, S, S) array per component.

The path kernels evaluate the likelihood and its statistics in one pass per lane block of
up to `local_selection.path_model.LANE_WIDTH = 64` samples (forward, then backward at once), so a fit keeps the
trajectories of one lane block per thread, not of every sample, in one scratch buffer per block search
(`workspace_bytes`); the release E-step accumulates carrier counts inside the kernel in batches of
`local_selection.path_model.RELEASE_SAMPLE_BATCH = 128` samples. Rows are deduplicated with a lexsort, never by
sorting a structured data type.

Every task reserves its planned peak memory with the run's memory governor before it starts. A block task
(`local_selection/rounds.py`, `block_task_bytes`) plans `local_selection.rounds.TASK_WORKSPACE_BASE_BYTES = 2147483648`
(2 GiB), the block's evidence slices and emission tables, the trajectory scratch for the largest panel the search
may reach (its start panel plus `local_selection.rounds.SEARCH_GROWTH = 8` rows, one lane block per thread), the
release batches, the candidate bank's discovery workspace (context rounds) and its likelihood memo with room to
grow. The pool separately holds `local_selection.rounds.WORKER_RESIDENT_BYTES = 805306368` (768 MiB)
per worker until it joins. A context refit (`local_selection/context_refit.py`, `refit_task_bytes`) plans its sites' slices, the
(N, B, S, S) emission array, each thread's (B, S, S) buffers, its outputs and 256 MiB for a worker's first task.
Blocks run longest first (by sites x states^2), so a round's tail holds the small blocks; cross-node bundles carry
the same estimates. The order and the pacing change only when work runs, never its result.

Completed blocks are recorded in chunks of 64, their memos before them; a context round's proposals
are recorded before its selection starts; a round is complete when its `selected` record exists.
An interrupted round resumes with its missing blocks, and a contig whose three rounds are recorded
returns the `l2` panels without decoding evidence or fits.

On the package's K-ladder simulation (N = 320, every block with K distinct founders) the initial round took 15.1,
32.1 and 60.8 CPU seconds per block at K = 12, 20 and 30 (worker peak 0.78 GiB at K = 30), round l1 0.4, 3.1 and
9.2, and a context refit 29, 80 and 217 per component ([performance](../performance.md#founder-counts)).

## Interpretation and limitations

- The selected panel is the best local optimum of a regularised objective reached by the search;
  it is not a posterior over panels, and K is not an estimate of the ancestral founder count.
  Within-block mosaics need not correspond one-to-one with ancestral haplotypes.
- Context rounds can only add candidate starts and change the dictionary code; they never add
  reads, and the incumbent is protected against a worse objective. Rare weak rows can still be
  lost when no candidate or move recovers them.
- Released alleles carry conditional one-bit support plus directional carrier evidence, not full
  allele posteriors. Unsupported alleles stay -1.
- The copying model uses the genetic map and one redraw intensity g for all samples and blocks; a wrong map or g
  changes how readily samples switch founders within a block.
- The unknown state absorbs copies no founder explains; a high `unknown_mass` flags a block whose
  panel misses haplotypes or whose reads are poor.
- Two rare founders can be replaced by one mixed row (release 2.17 diagnosis, f3 simulation chr2): discovery found
  one of them, the initial round dropped it, and a context round introduced a row about equally far from both (36
  of 200 sites from each, the two founders 75 apart). With fixed panels and frequency-only refits, the round's
  objective (path log-likelihood plus the context code) scores the mixed panel -807.0, the panel with the one true
  row in its place -1,122.0 and the panel with both true rows -987.3: the two true rows gain 44 log-likelihood nats
  but cost 225 code nats, and still 138 without any context, so the code's row cost, not only the context, keeps
  the mixed row. Such rows are rare (at most 1-8 non-truth released rows per simulated chromosome) but can make
  assembly swap two founders over a long range, which identity flags do not detect.
- At more than ten founders a context-refit site whose two found posterior modes differ releases nothing
  (multimodal abstention). Releasing the founders both found modes agree on is not safe: agreement of two local
  optima bounds no mass in a third basin (a fully enumerable 12-founder site where both modes agree on a founder,
  each with flip costs above 20 nats and 11 carriers, has exact posterior 0.875 for it). On K30 it would also gain
  almost nothing (2.17 lab: no new context-proposal calls and 7 more released cells of 227,598 in the next round).
