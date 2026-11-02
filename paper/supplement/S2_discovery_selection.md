# S2. Local founder discovery and panel selection

## Local founder discovery

**Inputs (DI01).** Blocks hold `block_sites` = 200 sites; discovery sites have total depth of at
least `minimum_total_depth` = 5 reads. Samples are sorted by name for the search and returned in
input order. A block is informative when some discovery site is observed in some sample. The ceiling
$\min(2N_a,2^M)$ limits births only; there is no other cap on $K$.

**Tie quanta (DI02).** Quanta are fixed once per block from its null NLL
$\mathrm{NLL}_0=\sum_i\sum_{j\,\mathrm{obs}}[-\log\max_g p_{ij}(g)+2\lambda]$, an upper bound on any
panel's NLL. Each is the smallest power of two at least
$\max(\text{floor},\text{factor}\times \mathrm{NLL}_0)$: NLL comparisons $2^{-10}$ and $2^{-20}$
($q_{\mathrm{nll}}$); A-step bands $2^{-13}$ and $2^{-26}$ ($q_{\mathrm{sample}}$); H-step allele
changes $2^{-12}$; improvement tests $2^{-10}$ ($q_{\mathrm{imp}}$); release cells $2^{-14}$
($q_{\mathrm{cell}}$); probabilities $2^{-15}$. Each floor is at least 16 times the measured motion
of its quantity under a $10^{-6}$ relative perturbation of the genotype likelihoods, so no decision
turns on numerical noise. Tie classes are formed by consecutive-gap single linkage; inside a class,
stated preferences (such as fewer founders) precede an arbitrary content order, except that value
order comes first in a class wider than four quanta. Budgets admit whole classes: a class straddling
a budget is admitted up to twice the budget and omitted whole beyond it (recorded); classes ranked
below a filled budget are ordinary truncations.

**Fixed-$K$ fit (DI03–DI09).** $\lambda$ = `unknown_copy_penalty` = 0.5 nats; $\tau$ =
`switch_penalty` = 10 nats; `snps_per_bin` = 10 (one site per bin in blocks of at most ten discovery
sites); likelihood floor $10^{-12}$; `max_iterations` = 50. The A-step band is
$[\min,\min+q_{\mathrm{sample}}]$; the category preference (row pairs, $(k,U)$, $(U,U)$) is never
reversed; the meet is order-free, e.g. $\{(k,k'),(k,k'')\}\mapsto(k,U)$. Samples without observed
sites take $(U,U)$. In the H step homozygous carriers count twice in usage and once in evidence;
unused rows keep their alleles; a comparison that an ordinary floating-point sum cannot certify is
re-summed exactly. The second usage-tie schedule runs only if the first meets a tie. A cycle returns
its lowest-NLL state (ties keep first-visit order) and is never reported converged. Each start
contributes its assignment-only solution and refitted endpoints; endpoints with duplicate rows are
refitted recursively at lower $K$. Panels are canonical (rows in lexicographic order), deduplicated
by content (a later fit replaces an earlier only if better by more than $q_{\mathrm{nll}}$) and
ordered by NLL class, smaller $K$, then content. Panels differing only at alleles no assigned
carrier observes are one archive entry. Memoisation and fixed-point certificates (DI07) change work,
not endpoints.

**Cavity score (DI10).** $\alpha$ = `state_concentration` = 0.5, $\omega$ = `unknown_prior_mass` =
0.02, $\delta$ = `double_unknown_fraction` = 0.25; base masses are $(1-\omega)/K^2$ for $(k,k)$,
$2(1-\omega)/K^2$ for $(k,k')$, $\omega(1-\delta)/K$ for $(k,U)$ and $\omega\delta$ for $(U,U)$,
renormalised. A held-out sample tied among states $\mathcal{T}_i$ removes $1/|\mathcal{T}_i|$ from
each. Mean-field ALT marginals $\mu_{kj}$ start from $h_{kj}$, $1-h_{kj}$ and $1/2$; rows without
remaining support stay at 1/2; a logit within $q_{\mathrm{nll}}$ of zero is set to zero, keeping
symmetric evidence symmetric; each start runs at most `mean_field_max_iterations` = 100 sweeps to
`mean_field_tolerance` = $10^{-10}$. Starts are compared by training evidence lower bound within
$q_{\mathrm{nll}}$; all converged winners (all starts if none converged) form connected classes at
$10^{-10}$; their predictive likelihoods, never their marginals, are averaged with equal class
weights before the `likelihood_floor` = $10^{-300}$ and logarithm. No held-out evidence sets these
weights. Unconverged solves are counted and block certification but remove no candidate.

**$K=1$ starts (DI11–DI13).** Similarity is $\bar s_{ii'}=M^{-1}\sum_j\sum_g p_{ij}(g)p_{i'j}(g)$,
distance $\max\bar s-\bar s$. Level-wise HDBSCAN* merges equal edges simultaneously and selects by
excess of mass (`seed_min_cluster_size` = 2). A cluster's mean ALT fraction $\tfrac12p(1)+p(2)$ is
hardened at 1/2; within $2^{-15}$ of 1/2 the population ALT fraction decides. Seeds enter by whole
cluster-size classes, largest first, up to `data_seed_count` = 8 (straddle 16); without cluster
seeds, dosage rows of the most decisive samples (most observed sites, then largest summed maximum
likelihood) are used. Fitted starts are kept by whole NLL classes up to `data_start_beam_width` = 6
(straddle 12); if every seed class is omitted the block is unresolved. Above `seed_landmark_budget`
= 256 samples, clustering uses at most 256 landmarks lifted back to all samples (DI12); fitting and
scoring still use every sample.

**Ordinary moves (DI14–DI16).** Deletions precede gauge rewires. Maximum cuts are exact per
component up to `exact_cut_max_k` = 12 rows, otherwise found by degree-ordered local search (a
recorded limitation); tied maximum cuts form one family with nominal budget $4\times$ `max_cut_ties`
= 16, admitted whole up to 32 and omitted whole above. Each cut yields a sitewise start (a site
flips only if the samples with both copies on one side gain more than $q_{\mathrm{nll}}$) and a
start flipping all eligible sites. Deletions and rewires are admitted by whole families up to
`max_proposals_per_expansion` = 128 (straddle 256). Data births rank samples, over both schedule
endpoints of the parent, by classes ($q_{\mathrm{sample}}$) of excess cost over the per-site
genotype oracle, then more observed sites. Each sample supplies, for both extremal dosage families,
subtraction seeds $\max(0,\min(1,\text{dosage}-h_k))$ for every row $k$ and direct dosage rows
(dosage 2, or dosage 1 or missing where the population ALT fraction exceeds 1/2, becomes ALT). Whole
tie groups are collected up to $\max(16,2(K+1))\times$ `data_birth_attempts_factor` (3) starts; a
group crossing `data_birth_tie_budget_factor` = 2 times that is omitted whole. Fitting stops at
$(K+1)\times$ `data_birth_children_factor` (3) distinct complete children; collapsed endpoints are
discarded; with no complete child, assignment-only fits are kept by NLL classes up to $K+1$.

**Soft-residual births (DI17–DI20).** Rows with at least `min_soft_responsibility` = 0.25 expected
copies in a sample (diplotype posterior with a uniform pair prior) are subtracted; the residual ALT
probability, given subtracted allele $h$, is $P(\cdot\mid h+1)/[P(\cdot\mid h)+P(\cdot\mid h+1)]$,
called at `residual_call_probability` = 0.8. Distances are Hamming fractions over jointly called
sites (distance 1 below `min_residual_joint_known_fraction` = 0.10). Groups of diameter at most
`max_cluster_hamming` = 0.10 are all HDBSCAN* tree clusters within the bound plus a greedy
edge-clique cover of the remaining records, budgeted per connected component,
`max_alternative_groups` = 64 (straddle 128). Pools are global and per (subtracted row, dominant
partner) route; a record joins every route whose partner mass is within the probability band of its
largest, so no convention selects one route. A consensus pools weighted log odds, one vote per
sample capped at weight 1, and is emitted with at least `min_unique_sample_support` = 2 samples,
effective support `min_effective_sample_support` = 1.5, a fraction `min_candidate_known_fraction` =
0.8 of sites called at `candidate_call_probability` = 0.9, and when it is no near duplicate
(`dedup_hamming_fraction` = 0.005 over `min_dedup_joint_known_fraction` = 0.6 joint sites, coverage
`min_dedup_bidirectional_coverage` = 0.95 both ways) of a panel row or better-supported emission.
Candidates enter by whole support classes up to 128 (straddle 256) and are hardened at 1/2 as for
$K=1$ seeds. Replacements delete any row of a birth, taking births by NLL classes up to
`max_replacement_children` = 48 starts (straddle 96).

**Archive, expansion and status (DI21–DI25).** Each $K$ keeps whole NLL classes up to `beam_width` =
6 (straddle 12); a wider leading class empties the beam while its last scored class stays
selectable. Every fitted panel is remembered: one that has left the beam re-enters only with an NLL
better by more than $q_{\mathrm{nll}}$. Refresh visits $K$ by $|K-K_{\mathrm{sel}}|$; new scores
count against $96+12K_{\max}$ (`exact_score_backstop_*`), and a class that cannot be scored whole
leaves the previous one in place. Score classes chain across $K$; in a class wider than four quanta
only the exact maximum survives; the smallest $K$ in the class is selected, so another $K$ can link
a class but never wins. All parents of a class are expanded before reselection. Replacements run
after a phase without improvement above $q_{\mathrm{imp}}$; bridges run in the residual phase when
$K+2$ is within the ceiling and $K+1$ is scored. Expansions stop at $32+8K_{\max}$
(`expansion_backstop_*`). Stop reason, omissions, certification, ambiguity, score margin (selected
score minus the best other $K$) and unresolved status are recorded and never steer the search.

**Release (DI26–DI27).** `min_call_probability` = 0.85, `min_supporters` = 2. A carrier contributes,
and counts once, only when its cost difference exceeds $q_{\mathrm{cell}}$ in magnitude. Tied
winners are matched by known content: a row absent from any alternative is withheld, and a call
survives only where all matching rows agree, with the least confident probability and smallest
support.

**Cost.** An A step costs $O(NK^2B)$; a trajectory iteration $O(NMK^2)$ (at most 50 iterations, two
schedules); the search $O(NMK^4)$ under exemption 0c ($K^4=810{,}000$ at $K=30$); a mean-field sweep
$O(NMK^2)$; release $O(NKM)$. Budgets are scientific parameters, independent of computing resources.

## Local panel selection

**Model (LS01–LS02).** $u$ = `unknown_mass` = 0.01; $\gamma$ = `redraw_rate_per_morgan` = 3, a
redraw intensity rather than a crossover rate. Recursions are scaled and samples are summed in
ascending order, so results do not depend on threads.

**Objective (LS03–LS04).** Context founders without calls are dropped; founders with equal rows
remain separate claimers. Among descriptions within a relative $10^{-9}$ of the shortest, the
largest matching is taken. Edit distances count called sites only, so one changed call moves the
code by at most one edit increment.

**Fit (LS05–LS07).** `tolerance` = $10^{-6}$ nats. Rows whose summed gains lie within a relative
$2^{-10}$ (`FLIP_TIE`) of the largest are compared by the exact objective after flipping; if
different panels remain tied the fit stops. Updates stop at $20+4K+2M$ (`update_backstop_*`); rows
are returned in canonical order. The trajectory-best state is the earliest strictly best; its
frequencies are then refitted while the context objective rises. A snap proposes each called site
where a matched row differs from its founder, unless the flip would duplicate a row, with bound
equal to the exact one-bit gain plus the edit saving $\log V(n,d)-\log V(n,d-1)$. It needs a bound
above $\eta$; bounds within $\eta$ of the largest form one class, decided by the smallest canonical
panel; a snap is kept only if the objective rises.

**Search (LS08–LS15).** Deletions are proposed in row order, merges in lexicographic pair order and
additions in bank order (new row frequency $1/(K+1)$); a single-row panel has no deletions or
merges. A merge's consensus row takes the smaller row's allele within $2^{-20}$ of 1/2; the merged
row carries $f_k+f_{k'}$, and a consensus equal to another row merges that row too. Starts with
equal panels and frequencies within $10^{-12}$ are one start. The merge bound is the deletion
panel's log-likelihood plus the kept row's expected gains at changed sites plus
$\sum_k\bar n_k\log(f^{\mathrm{m}}_k/f^{\mathrm{d}}_k)$ ($\bar n_k$ and $f^{\mathrm{d}}$ from the
deletion panel, $f^{\mathrm{m}}$ the merged frequencies on its rows), the larger of two such bounds.
Exactly scored merges are, per row, the best-bounded merge with a nearest row and the best-bounded
merge not involving it whose consensus is nearest it. Each kind refits whole classes until
`refits_per_kind` = 8; a class crossing `refits_per_kind_cap` = 16 is excluded. Refits reaching one
panel are one candidate. Rounds stop at $10+2K$ (`round_backstop_*`). If a start's fit reaches the
previous round's panel, the previous fit is kept unless beaten by more than $\eta$; if the previous
fit is better by more than $\eta$, the search starts from it. A whole search is reused only for
identical inputs; fits, scores and bounds are memoised across the first two rounds (LS13, changing
work only). Outcomes of several starts are compared by objective class, then fewer founders;
distinct tied panels become alternatives (LS14). A retained incumbent discards the attempted
alternatives (LS15).

**Bank (LS16–LS20).** The initial round has no incumbent and no bank search. A proposal row with no
call is dropped; a partial row takes, at uncalled sites, the alleles of every latent row at minimum
Hamming distance; tied completions give one source per combination up to `completion_variant_cap` =
8, above which the source keeps only uniquely completed rows. A source's best set is the nested
class of cavity score, then NLL, then fewest founders; each member is released by discovery's rule,
with uncalled cells taken from the source's own latent panel. The bank search first fits the whole
bank as one start, then visits deletions, additions and row replacements (closest first) with
$\kappa$ = `complexity_scale` = 0.5, reselecting by nested classes of penalised score, NLL and
founder count, and stopping without improvement above $q_{\mathrm{imp}}$ or after
$8+\lceil K/2\rceil$ rounds. It fits at most $512+64K$ novel starts per call and admits at most
$1024+128K$ panels, overlapping fit families entering by whole NLL classes (LS20). The search bank
is the latent rows, starts, selected panels' rows and the previous panel's rows.

**Release (LS21–LS24).** `min_call_probability` = 0.85, `min_supporters` = 2, `favourable_margin` =
$2^{-14}$ nats. Carrier probability counts a homozygous state once; a carrier whose own gain lies
within the margin of zero is not favourable. A previous release is reused only for identical panel
and frequencies without context witnesses. A site's unknown-copy count is the smallest count whose
mass is within a relative `unknown_count_band` = $2^{-15}$ of the largest; samples without depth
report 2; tied alternatives take the maximum. Rows shared by all alternatives keep agreed calls,
mean ALT probability and smallest support.

**Context refit (LS25–LS28).** `context_levels` = (1, 2). An uncovered block receives its current
calls without context founders, hence no witnesses, claims, trajectory-best adoption or snaps. A
component must match its blocks' concatenated positions exactly; duplicate component rows give a
consensus template. Bins are blocks, spaced by the map interval between block centres; founders have
prior mass $(1-u)/K$ and the background state $u$. Uncalled founders and the background carry the
site's ALT frequency $(1+n_{\mathrm{ALT}})/(2+n_{\mathrm{called}})$ over called founders; the
emission mixes the expected likelihood with $1/3$ at weight `emission_mix` = 0.01. A site's
founder-allele posterior, under a uniform prior, is enumerated exactly for at most 10 founders;
above that, a belief-propagation mode and local ascent fix certain founders (flips costing over 20
nats), withhold multimodal sites, and enumerate the remaining core exactly (at most 10 founders) or
within a Hamming ball of at most 1,024 configurations (radius at least 3), with a shell guard. Calls
are released at `posterior_threshold` = 0.99 with count at least `min_support` = 2 or mass above
$2+2^{-20}$; other cells keep the component's call, then the template. Duplicate local rows sum
support and record multiplicity. Recorded summaries do not affect decisions (LS29–LS31).

**Cost.** A forward–backward pass is $O(NMK^2)$; a fit at its backstop $O(NMK^2(K+M))$; a climb
$O(NMK^3(K+M))$ under exemption 0c; context-coded screening $O(K^5)$ per round under exemption 0
($K^2$ merges each with an $O(K^3)$ matching; $24{,}300{,}000$ at $K=30$); snaps $O((K+M)^2)$
passes; context refit $O(NBS^2)$ time and memory per component.
