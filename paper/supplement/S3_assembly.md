# S3. Assembly, founder release and refinement

Parenthesised identifiers are the software's method units; defaults are those of release 4.4.0 (none is a user setting). Notation: $T$ bins of a batch; $W$ units of a proposal window; $b_w$ surrogate bins of a window; $R$ the window's row-alphabet size.

## Founder completion

**Folds (U03).** `fold_seed` (1729) is hashed with each sample's observation-mask content and depth key; equal-key groups, visited in byte order, go whole to the lighter fold (ties and inactive groups to fold zero). No random draw or label chooses a fold.

**Cross-fitted variational EM (U04).** Samples copy unordered row pairs and unresolved alleles are binary variables shared across samples, both under uniform priors; emissions are $\max\{\log(0.99\bar p+0.01/3),-2\}$. The variational family factorises over samples (pair states) and sites (configurations). A site's support is the full cube up to `MAX_EXACT_BITS` (10) unresolved founders. Above that it is an uncertain core fixed from the first iteration, so the ELBO stays a lower bound. One start runs 2–100 iterations (ELBO tolerances $10^{-8}$; a decrease beyond $10^{-8}$ is an error). Training-fold calls need posterior ≥0.99, ≥2 carriers, non-exchangeable founders, a unimodal site and a passing shell guard. The held-out fold is painted only where the released panel calls every founder; absent training data leaves the pre-fill panel (recorded).

**Cavity fills (U05).** Bins hold 50 SNPs (`snps_per_bin`), and outside-bin statistics are prefix plus suffix sums. A founder is ineligible if another founder has an identical outside-bin allele vector (tolerance $10^{-12}$) or differs from some founder at fewer than one observable outside-bin site. Support, over focal-informative samples sensitive to the target, is the larger of the number with leave-bin-out carrier probability above 1/2 and the sum of those probabilities. A fill needs support ≥ 2 and posterior ≥ 0.99. A single-bin multi-founder block therefore gets no fill.

**Site posteriors (U08–U10).** Dosage polynomials are enumerated exactly; a value cancelling below $10^{-10}$ is recomputed as a direct positive sum. Above ten unknown founders, damped loopy belief propagation (damping 0.5, tolerance $10^{-9}$, ≤500 iterations) [@murphy1999] is followed by deterministic ascent from its mode and complement (backstops 50), on an uncertain-core cube or a Hamming ball (≤1,024 configurations, radius ≥3, 3-nat shell guard). Non-converged BP is flagged; this can withhold a fill even when a mode exists.

**Boundary links (U06).** Rows with identical pre-fill vectors form classes. Two classes across a boundary are linked only if, on samples informative on both sides, the mean squared discrepancy of expected copies and of $\min(\text{copies},1)$ is ≤0.099; the pair is a reciprocal unique minimum by margins 0.03 (pooled) and 0.01 (each fold); there are ≥6 pooled and ≥3 per-fold informative samples; each class has ≥1.0 pooled and ≥0.5 per-fold effective copies and one or more anchor sites separating it from the other classes of its block; and both blocks have two or more classes. Exact best/runner-up ties are not broken by row number.

**Occupancy gate (U07).** A block releases nothing if selection left unexplained copy mass above `maximum_discovery_unknown_mass` (0), or if it has a single founder.

*Abundant route:* squared carrier probabilities sum to ≥8 and ≥5 samples carry the row with probability ≥0.8; all its candidates are released (a sum within the tie band of 8 is not abundant). *Rare-continuity route:* through an accepted single-row link, ≥2 samples informative on both sides carry the row (≥0.9) and the neighbour (≥0.8), product ≥0.72; the best neighbour must be unique (tied neighbours abstain only when their qualifying samples differ); fills are released only where a qualifying sample is focal-informative.

Released fills publish the allele with posterior above 0.5.

## Hierarchy

**Partition (U11).** Linking proxies stride to at most `max_sites` (2,000) usable sites per component. A segment of $n$ components is cut into $m=\lceil n/10\rceil$ batches, plus one when $n$ is odd and $m$ even. Sizes are near-equal, with the larger ones in mirrored pairs. Breaks persist. A batch with one component or no common informative sample passes through unchanged.

**Levels (U02, U12).** Stop reasons, checked on unrefined join output, are in order: single component; no reduction (level ≥2); only phase breaks (level ≥ `minimum_levels`, 4). Joined founders are sorted lexicographically, so later slot tie-breaks are content rules. Final-role refinement follows every level, with additions only at the level whose stop reason is set. First-level paths, refined first and then unrefined, form the coarse alphabets of later refinements. Local selection runs one or two unrefined levels as context.

**Alternatives between levels (U13).** A family is an input choice only if it covers a component's exact span and contains that component's nominal panel; otherwise it stays unresolved. If the product of choices is at most `BRANCH_LIMIT` (16), every joint branch is joined on the sites usable under all choices and one common cohort; each count's winners are its top LL tie class, and the representative has the smaller $K$, then fewest exact switches, then content order. A larger product is recorded as an unresolved joint budget; a tied family is never reduced to a convenient subset.

## Linker and beam

**Scans and EM (AJ05–AJ07).** Emissions are $0.99\Pr(\text{obs}\mid\text{class})+0.01/3$ over seven predictive pair classes, without a floor. Inside a component a homologue switches with $\theta$ equal to the interval's map distance in Morgans (bp times the fallback rate without a map), capped at 0.5; weights $(1-\theta)^2$ (stay) and $\theta(1-\theta)/(K-1)$ per other founder are normalised. An error-tract chain has stationary fraction 0.01 and mean tract 20 kb; error-state emissions are the mean of the three likelihoods. EM per gap starts uniform; the M-step adds 0.1 pseudocounts, floors logs at −10, renormalises and mixes 0.01 uniform; it stops at $\max|M(\mathbf A)-\mathbf A|\le10^{-3}$ or after 50 full updates (recorded as unconverged).

**Horizon and beam (AJ08–AJ11).** The horizon is $G=\max\{1,1+\lfloor0.5/(3\bar D)\rfloor\}$ (`recombination_tolerance`, `gap_horizon_scale`). The beam builds paths forward with 16 per endpoint row, then refines block by block backwards, and archives every selected path. With constant transitions (AJ10, retained accretion), only $O(BK)$ first-covering paths are proposed. Linkage (summed step scores) is consulted only within LL tie classes.

## Join panel search

**Objective (AJ01–AJ04).** The bin size is $\max\{\min(100,\max(10,\lfloor\bar L/20\rfloor)),\lceil L_{\mathrm{span}}/2000\rceil\}$, with $\bar L$ the mean number of sites per input. The switch source is the first maximiser in pair order, and a state stays on equality. The exact minimum switch count is only a secondary prior. Proposals come from tracebacks under both conventions.

**Start and tier 1 (AJ12, AJ15, AJ16).** All anchor panels are scored. Proposal kinds run in order: replace, static replace, novel replace, local, one-sided, splice, cycle, switch-splice, switch-cycle. An exploratory painting at penalty $\min(\lambda,10)$ exposes coordinated switches for proposals only. At most `full_scores_per_kind` (16) edits per kind are scored exactly (a cut class kept whole, up to twice that). A move improves when $2\Delta\mathrm{LL}>\max\{10^{-8},10^{-9}(1+2|\mathrm{LL}|)\}$.

**Admissibility (AJ14).** A move must keep $K$ distinct paths, and each changed path must keep a private cell and stay more than `near_identity` (0.01) from every other path. Admissibility is checked on the edit witness, before deduplication; a seed and its copy are exempt during their refit.

**Bounds and tier 2 (AJ17–AJ20).** Exclusion forwards, the maxima without up to two paths, are recomputed only where those paths cover a bin's maximiser set; this is exact under ties. Two-region bounds limit removals of up to two paths with additions of up to two. Only a certifying bound skips work; samples whose bound exceeds their base by $10^{-9}(1+|\cdot|)$ are scored exactly. Tier 2 enumerates pool replacements and the local, one-sided and splice moves in canonical order, scores them best-first and records flat neighbours.

**Pool, plateaus, replays (AJ26, AJ28, AJ29).** Quotas scale with row multiplicities. The quota ladder doubles to 256, and to 1,024 after an acceptance at 256. A pool with identical ordered content triggers no count work. Plateau continuation follows uncertified winning-family members. Up to `MAX_BRANCHES` (4) searches per join, the primary included, replay forced choice prefixes (FIFO); a mismatch or exhausted budget records incomplete resolution, and seed choices are never forced.

**Backstops (AJ25, retained accretion).** With $K_b$ the founders of input $b$, a search stops after $100+20\max_bK_b$ sweeps. Births and seed swaps freeze after $50+10\max_bK_b$ count changes, while deaths continue. Backstop stops are reported as uncertified; a final census of paths failing their own death test is reported only.

## Join count step

**Deaths and births (AJ21, AJ22).** Paths without novelty are deleted one by one, starting with the largest $\mathrm{LL}(\mathcal P\setminus f)$. Otherwise the best candidate is ranked by that value in wide tie classes, then by negative linkage, then by content, and is tested. A birth is scored only if it passes two necessary conditions on upper bounds $u_i\ge d_i\ge q_i$: the second-largest $u_i$ exceeds $c/(2\tilde N)$, and $\sum u-\max u>c/(2\tilde N)$. Survivors are scored exactly in decreasing bound order, until no bound can come within four quanta of the best accepted gain.

**Duplicate seeds (AJ23).** When ordinary births fail but a pure-mosaic pool path supplies a chimera indicator, every selected founder is duplicated in turn. Every seed is refitted by certified moves restricted to it and its copy, then settled (paths without novelty deleted, then a death test). A refit competes if it survives, is not smaller and raises LL; acceptance is a birth or a seed swap at unchanged count. Endpoint LL, its tie class and content order decide among survivors. Indicator and coverer ranks do not restrict seed admission.

**Cycles (AJ24).** Born-path tags follow same-count edits; deleting a tagged path freezes births, and is marked undecided if ambiguous in a cycle that began with a decided deletion. Revisiting a canonical count state freezes births and seed swaps.

**Bootstrap and shadows (AJ30–AJ33).** Bootstraps draw 2,000 resamples (level 0.95) from a stream keyed by content (decision kind and larger panel in joins; component end positions, size, round and kind in refinement), shared across a shadow family. Bands are resolved only for reported decisions. A complete shadow family of up to 2,048 members is enumerated only if the total test passes. In joins, an unverified count remains a diagnostic.

## Proofs and bound tests

**Count bounds (AJ34, RC10).** Optimistic scores let each block choose a free panel, resetting between blocks. A block with $K+1$ rows takes its best one-row deletion. Deletion bounds need every founder to have a private cell. With $\underline d_i=F_i(\mathcal P_K)-U_i$ ($U_i$ the per-sample bound), a deletion round is skipped when $\sum\underline d/c-\tfrac12$ and $(\sum\underline d-\max\underline d)/c-(\tilde N-1)/(2\tilde N)$ exceed the rounding tolerance and the replicate-wise percentile of $\sum\underline d/c-\tfrac12$ is positive.

An addition round is skipped if the total bound gain $U-F$ is below $c/(2\tilde N)$, or if every $u_i=U_i-F_i$ is (which also proves non-ambiguity). Skips are recorded as proven decisions.

**Shadow-envelope certificate (AJ35; retained accretion).** In refinement only, a proof is attempted for an over-budget family when the largest per-block product is ≤2,048 and local work ≤4 times ordinary work. Block messages maximised over local products give $U^{\sigma}_i$ bounding every shadow, so $q_i\ge F_i(\mathcal P^+)-U^{\sigma}_i$. The proof succeeds only if every combined field equals the incumbent's under every permutation of each bootstrap draw. A successful proof clears the count-unverified flag only.

## Refinement: objective and fixed-K search

**Workspace (RC01, RC02).** Objective sites are kept sites with a called prepared row; at an all-uncalled site, pairs would differ only through the allele prior. Samples without evidence contribute exactly zero and are excluded from $\tilde N$ ($F$ still sums all samples). Interval hazards are `reset_rate` (1 per Morgan) times map distance, floored at $-\log(1-10^{-15})$ and summed over skipped sites; the first informative site has no transition; costs are stored in single precision. A map is required; genotype error is fixed at 0.01. The recursion over unordered pairs is

$$\mu'\{a,b\}=\max\big(\mu\{a,b\},\;\max(\bar\mu_a,\bar\mu_b)-\kappa,\;\textstyle\max\mu-2\kappa\big)+\epsilon(a,b),\qquad \bar\mu_x=\max_y\mu\{x,y\}.$$

Exact ties prefer lower switch cost; a separate recursion counts switches (RS02).

**Surrogate and windows (RC03, RS04).** The bin size is $\max\{\text{floor},\lceil L_c/2000\rceil\}$. The floor is $\max\{200,\lceil L_{\mathrm{chrom}}/2000\rceil\}$ at levels 1–2 and 200 thereafter (`proposal_min_sites_per_bin`, `proposal_max_bins`). The Potts penalty is 20 nats. Fine windows of 8 blocks at stride 4 offer all prepared rows. Coarse windows of 4 first-level spans at stride 2 offer the distinct first-level paths.

**Window programme (RS05, RS06; exception 0e).** A box bound certifies already-optimal windows. Otherwise an exact frontier programme prunes by closed-message dominance ($10^{-6}$) and valid bounds. Past a frontier of 256 it restarts with Lagrangian suffix bounds (≤100 dual sweeps) [@rush2012] and a feasibility beam of 16, keeping ≤512 children per unit; it is approximate where that cap binds, and such units are counted. A frontier beyond `frontier_bytes` (128 GiB) keeps the incumbent. The canonical answer is the incumbent when it lies within $2^{-14}+10^{-9}|H|$ of the optimum ($H$ the incumbent's surrogate value), and otherwise the lexicographically first path within that band.

**Work account (RS07, RC18; accretion, exception 0e).** Each template is allowed $2^{25}NWb_w^2\max(K,R)^3$ charged units (`window_work_coefficient`). Exhaustion discards the template and stops the component, which keeps its input rows and families plus an unperformed-search family; during polish it keeps the unpolished panel. The bound is per template, not per component.

**Acceptance (RS08–RS10).** At each junction, exchanges come first, at most 4 (`EXCHANGE_ROUNDS`). Content moves follow, fully re-ranked through 4 acceptances (`CONTENT_FULL_ROUNDS`) and then rescored singly. The threshold is TOL = $2^{-9}$ nats. A coarse move with $\Delta F+\Delta(\text{switch cost})<-10^{-6}$ is guarded: affected paths are traced with and without it, switch events are moved to the leftmost site of their evidence interval, paired excursions are excluded, and the move needs $\Delta F-C_{\text{shared}}>\text{TOL}$ with $C_{\text{shared}}=\sum_i\min(\text{saving}_i,\text{cost of shared one-way removed events})$. Fine moves and bridges bypass the guard.

**Fits (RS11, RS12).** A fit converges at a sweep that accepts nothing. Reaching `sweep_limit` (100) records nonconvergence. An $F$ that deviates from its prediction by more than $10^{-9}\max(1,|F|)+10^{-6}\times$ (accepted moves) is an error. Of the two direction starts, the winner is the higher $F$ class, then the minimum switch count; tied variants are retained, and rows are canonicalised.

## Refinement: count pass

**Rounds and candidate refits (RC09–RC11).** Refit the incoming panel and run ordinary deletion rounds. Candidates are all founders without a private cell when any exist, otherwise all founders; every eligible candidate is refitted. At an addition-enabled level every founder is a duplicate seed, and every seed is refitted. Optimistic bounds may certify an entire round unneeded. Compare converged endpoint $F$ with the incumbent's scale, choose the content-smallest origin in its best tie class, then evaluate that class's verdicts. Deletion stops on the first rejection; additions stop on rejection or at twice the incoming $K$. Every refinement deletion uses the ordinary count rule.

**Symmetric null and ties (RC12, RC13).** The death class is the top tie class of unrefitted $F(\mathcal P^+\setminus f)$, taken over founders without a private cell if any exist. Every member is evaluated with one bootstrap stream; disagreement or an unresolved member family yields $\mathcal P^-$ with ambiguity. Every member of the selected refit's tie class is also decided. Opposed verdicts keep the smaller-panel outcomes, which are ranked by margin, then by fewest switches.

**Restoration, terminal family, continuation (RC15–RC17).** A fit is remembered at its $K$ only if it improves by more than $\max(\text{TOL},\text{quantum})$. At the final $K$ a decisively better fit is restored and its adjacent-$K$ comparisons rerun. The returned panel's $F$ class is rescored exactly; its fewest-switch member represents it and the live family is released. Continuations replay alternative input roots and count choices breadth-first from the original input, within `alternative_branches` (16) leaves. Winning classes are compared per count, preferring the smallest $K$, then the fewest switches. Overflow stays unresolved.

**Polish (RC19, RS14–RS17).** Polish applies one-direction fits to the decided panel and to each distinct live alternative. At each ordinary fixed point it tries a beam escape (width 64); if nothing is accepted, a long repair (one-founder moves over 64-block windows at stride 16, beam 16, each with an optional 8-block private correction from the two most-agreeing donors or the incumbent); if still unchanged, a bridge (fine precursor plus suffix exchange). Accepted gains are verified exactly; counts are not re-evaluated.

## Ties

Each quantum is the smallest power of two that is at least $\max(\text{floor},\text{factor}\times|\text{scale}|)$:

| Kind | Floor | Factor |
|---|---|---|
| `join_ll` | $2^{-19}$ | $2^{-34}$ |
| `join_ll_wide` (deaths) | $2^{-19}$ | $2^{-30}$ |
| `join_proxy` (bounds) | $2^{-19}$ | $2^{-24}$ |
| `beam` | $2^{-18}$ | $2^{-30}$ |
| `refine_F` | $2^{-9}$ | $2^{-26}$ |
| `count_margin` | $2^{-26}$ | 0 |

A class chains values whose consecutive gaps are at most one quantum, and a class wider than 4 quanta keeps value order (U14, AJ27, RS09). Within a class, linkage decides first (joins only), then content. Convention reversal changes only arbitrary content orders, never a prior.

## Founder release (U16)

Alleles that no painted carrier observes are withheld first. For each retained family, blocks whose row multisets differ are isolated and their unmatched rows withheld; greedy forward and reverse scans cut continuity wherever the panels stop agreeing up to a row permutation. Segments are reordered by content, and withheld rows are barred from the downstream r5 fill. Count-unverified records annotate segments without withholding. A family not matching the final panel is reported as unresolved correspondence.

## Cost class in K

| Mechanism | Dominant cost | Class |
|---|---|---|
| Completion | $O(NLK^2)$ cross-fit; $O(NK^3)$ ball sites; $2^{10}$ cube | ≤ $K^3$ |
| Linker | $O(NJK^2)$ scans ($J$ emission sites); $O(NK^3)$ propagation | ≤ $K^3$ |
| Join search | Adaptive structural screen has a cubic analytic bound; direct query count has a quartic trend on a finite grid | cost register 0e (AJ17) |
| Count refits and shadows | $O(K)$ fits, $O(NLK^3)$ per round; repeated deletion rounds $O(NLK^4)$ worst case; ≤2,048 shadow scores per member | Complete eligible candidate domain; other search factors separate |
| Refinement search | $O(NLK^2)$ messages; $O(BNK^3)$ exchanges; window fixing bounded by the work account | exception 0e (RS05, RS07) |
| Release | $O(Bn_{\mathrm{alt}}K\log K)$, $n_{\mathrm{alt}}$ alternatives | ≤ $K^3$ |
