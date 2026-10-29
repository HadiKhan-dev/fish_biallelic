# Changelog

## 2.17.0 (29 October 2026)

This release measures where the remaining founder errors come from, reports what the data do not resolve instead of
hiding it, adds the validation tools that 2.16 lacked, and bounds local selection's merge screen at large K. Local
selection's search changes (one exact double-removal merge per nearest-distance class, below), so panels can change
where a distance class held several merges; the other inference stages are unchanged. The release kernel manifest
covers the new relative-proxy kernels. New are a structural founder
identity diagnostic with every export, a decision-free relative-proxy table for released parent edges, truth metrics
that separate identity from local allele accuracy, and explicit pedigree designs for simulation (with sample
metadata, so cohort eligibility and opposite-sex pairs get a truth test).

**Founder identity flags** (`export/identity.py`, `export/founders.py` `write_identity_flags`;
`OUTPUT/export/<contig>.founder_identity.tsv`; `docs/methods/founder_identity.md`, Identity flags). Where a founder row
copies other rows' prepared local rows on both sides of a junction with disjoint partner sets, no local content of its
own carries it across (`local_duplicate_partner_change`). Each flag names the exported column, its source component and
row, the block boundary with its flanking marker positions, the local rows and partner columns of both sides, and
whether the junction is also a release-segment boundary. Computed on the exported panel's painted row paths (after
label continuity) and named through the release segments. Diagnostic only: calls, columns and breaks are unchanged
(re-exporting the same 2.16 inputs gives a byte-identical founder VCF on std chr10 and chr20).
- Measured with the new identity metric (below) on the 2.16 runs: std (8 chromosomes) 1,487 wrong continuity
  assertions; flags cross 1,164 of them and 316 correct ones (0.36% of correct), multi-copy coverage 0.9954 -> 0.9951
  for a consumer that drops flagged continuity. K30: 87 of 146 wrong, 1 correct. f3 7 of 43 and backcross 16 of 34
  wrong (their errors have other causes). On a design whose founders are all well represented, about 20 flags per
  chromosome, nearly all on true shared stretches: a structural flag, not an error detector.
- Why identity errors remain: on std they appear mainly with the founder refinement of assembly levels 3 and 4, not at
  the joins (f3's appear by level 2; its level-4 jump in whole-component errors is the matching span, not new switches);
  most sit in one or two mosaic rows per chromosome that follow other founders' local rows, and the refinement objective
  prefers those panels to substantially more truth-like routings of the same local rows (std chr10 level 4: 2,156
  nats). More search does not remove them; shared-crossover accounting is not pursued.

**Local selection: bounded double-removal merges** (`local_selection/search.py` `double_removal_merges`;
`docs/methods/local_selection.md`). The merge screen scores exactly, besides their EM bounds, the merges of each row
with its nearest rows and the merges whose consensus is nearest a row. Until 2.16 every member of these
nearest-distance classes was scored exactly, up to K(K - 1)/2 forwards per screen (Hamming distances are integers, so
the classes are large at K = 30: a census found 76-277 of about 400). Now each class is represented by its member with
the largest EM bound (tie classes of the decision tolerance, then content; counted as
`ls.rank.merge_representative`), so at most 2K merges are scored exactly and the likelihood screening is O(N L K^3).
Likelihoods, objective, refit caps and stopping rules are unchanged; the other members keep their bounds. The context
regulariser's founder matching per proposal is not bounded by that figure (pre-existing). Measured on saved 2.16
round inputs with both searches run in full (95,253 blocks of the std, f3, backcross, K30 and real TroMau
designs, three rounds each): every simulated block and every context round is unchanged; 9 initial-round blocks
(2 K30, 7 TroMau) end at a lower objective (0.9 to 40 nats; on K30 one more true founder recovered in each), and
when each round feeds the next, 7 of the 9 return to the 2.16 result by the last round (TroMau chr14 one block
11.7 nats lower, chr6 one block 2.6 nats higher). K30 needs 51-57% fewer exact likelihood scores per round.
Reversing the tie conventions changes the same 6 of 600 K30 block-rounds before and after the change. End to end (fresh runs
through founder refit, rounds feeding rounds and assembly re-run): the std, f3, backcross, TroMau-like and a held-out
std seed are identical to 2.16 at every stage. K30 changes on one chromosome (m2: the changed initial block reaches
assembly level 2; its founder panel calls 7,214 more alleles with 570 more errors, wrong-or-missing alleles 55,725 ->
55,763 over K30, wrong continuity assertions 28 -> 46 against about 2,400 correct ones). Real TroMau keeps every
pedigree call; its paintings change on five chromosomes: on chr3 four founder rows are routed differently over the
first half (5.8% of panel cells after row matching; the paintings need 8% fewer switches), on chr14 3,170 cells of one
row (2% more switches), on chr2, chr5 and chr6 a few cells.

**Truth metrics** (`evaluation/identity.py`, `evaluation/evaluate.py`; `docs/simulation.md`, Founder metrics).
`haplotypes evaluate` adds to every founder stage: copy-count strata (truth alleles whose founder no sampled homologue,
one or several carry), and continuity assertions on a fixed 100-marker truth grid (identity switches, direct and across
unlabelled windows, by copy stratum; a window's interval ends at its middle marker; coverage of represented, distinguishable founders; assertions crossing the
stage's identity flags and the coverage left without them). The counts are read together: splitting rows can raise the
correct share without raising coverage. With metadata eligibility, the pedigree counts cover the eligibility's targets.

**Explicit pedigree simulations** (`simulation/design.py` `read_pedigree_table`, `simulation/pedigree.py`
`simulate_explicit_pedigree`, `simulation/source.py`; `pipeline/driver.py`, `pipeline/samples.py`, `config/load.py`;
`tools/cross_pedigree.py`, which rejects impossible mating-pair counts and fills a sex's shortfall of sequenced F1
from the other sex; `docs/simulation.md`, Explicit pedigrees). `[simulation] pedigree` takes a table of samples,
parents, sexes, generations and who is sequenced (founders may be); such a design may carry `[samples] metadata` (the
same table; `.tsv`/`.csv` tables are now read as metadata) and a cohort eligibility policy. Random designs simulate
byte-identical data (chr19/chr20 reads and truth digests equal to 2.16; the simulation's code identity changes, so a
2.16 simulation is regenerated, identically, on resume). A TroMau-like cross (2 dams, 1 sire, 34 F1 with
16 sequenced, 96 F2 from 22 matings, 58 of them without a sequenced parent; 8 chromosomes): 2.16 inference calls every
target exactly (96/96, 48/48 edges, no false edge; the sequenced F1 are the unsequenced parents' sibs), founders 2-107
wrong-or-missing alleles per chromosome.

**Relative-proxy contrasts of released edges** (`pedigree/relatives.py`, `pedigree/stage.py`;
`OUTPUT/pedigree/relative_contrasts.csv`; `docs/methods/pedigree.md`). For every child-candidate edge of the Tier-B,
Tier-B partial and complete views: the production M1 score minus a second-degree and a full-sib relationship proxy,
summed over the chromosomes where the candidate has evidence, with negative-chromosome counts. Uncalibrated, decision-
free. TroMau-like: every released (true) edge beats the second-degree proxy by >= 12.6 nats and the full-sib proxy by
>= 23.5; real TroMau: 2 of 38 released edges negative against the second-degree proxy (the proxy has the larger composite
score under these paintings; this identifies no relationship). Small positive contrasts also occur for non-parents
(K30: 2.4-3.3 nats), so only large positive totals separate.

**Measured and left unchanged:** TroMau's fitted evidence scale (0.10) changes no Tier-B call against 0.5
or 1.0 (it withholds one complete-view call, whose candidate is an F1 sib of the child's likely unsequenced
parents; no parentage truth exists for it); discovery's winning-set mean-field solves at
20x the iterations change no K30 panel or count; doubled local-selection bank backstops change no K30 panel;
per-founder release at multimodal context-refit sites is unsafe (an enumerable counterexample) and would add almost
nothing on K30; local selection can replace two rare founders by one mixed row (documented). The contamination QC is blind at about
1,000 samples: a private allele survives the one-read carrier rule with probability (1 - e)^(d (N - 1)), nil at that
size; an N-scaled carrier rule restores detection in simulation but, with the current event statistic, falsely flags
clean low-depth or sparsely observed libraries in dense-marker controls, so the QC is unchanged in 2.17.

Documentation: the 2.16 docs-check error (pedigree.md NOISE_FLOOR literal) is fixed.

## 2.16.0 (28 October 2026)

This release continues the pedigree work of 2.15. It removes a measured source of chromosome-scale penalties against
true parents: the parent model conditioned the alleles that the founder panel had withheld on the parent's own reads,
which were noisy exactly there in the traced simulations, and those few sites cost the traced true parents up to tens of
nats per chromosome against "no observed parent" and against relative models. With that fixed, the std and
missing-parent simulations release every true parent in Tier B. The direction evidence now respects that a parent's
junctions on one homologue are inherited together; resampled fits whose family messages stall get the damped retry
early; and the pedigree arrays that scaled with N × N for every dyad now follow the eligible edges. Parents versus close
relatives in inbred or low-diversity designs remains the main limitation, now measured without that artefact.

**Named founder classes transmit their panel alleles** (`pedigree/projection.py` `_transmitted_alt`;
`docs/methods/pedigree.md`, Transmitted-state projection). Where the founder panel left a named class's allele uncalled,
the transmitted ALT probability started from BACKGROUND's ALT frequency and was then conditioned on the candidate
parent's own GL row. The panel withholds alleles where the data are noisy, and in the traced examples the parent's reads
were noisy there too: its genotype argmax was wrong at 41-49% of such sites against 5-8% at fully called sites, and at
truly homozygous such sites the conditioned allele scored the transmitted one worse than the background frequency
(summed log predictive probability −96 against −65 on std chr13). M1 and M2 used these values for the child's emissions
while M0 did not, so these sites (0.3-1.8% of the selected markers) accumulated losses for true parents, directly and
through the filter: std F2_11 <- F1_7 chr13 M1 − M0 = −20.5 nats, F2_6 <- F1_6 chr19 −53.8, missing-parent F3_1 <- F2_10
chr2 −23.9, where the called ancestry and alleles of parent and child were mostly correct. Named classes now transmit
their panel allele, or BACKGROUND's ALT frequency where it is uncalled (the value the null and the painting use); only
BACKGROUND, which has no panel haplotype, keeps the read conditioning. The parent's reads still shape its source
posterior and bridges. On the traced chromosomes M1 now beats M0 by 3.5 to 44 nats. Found by a per-bin trace of the
forward likelihood and confirmed by an independent review, which also showed the enrichment is not explained by depth or
true genotype. It is a conservative treatment of panel-withheld alleles, not the generative likelihood.

- Pedigree replays on the saved inputs, truth from the simulation manifests: std Tier B 39 -> 40 of 40 exact, true
  edges 59 -> 60 of 60, wrong state claims 1 -> 0; missing-parent 44 -> 45 of 45, edges 50 -> 52 of 52; N = 320,
  backcross, f3, f3 on 8 chromosomes and TroMau (0.01-0.03% uncalled alleles): no call change. K30 (30 founders,
  low diversity): Tier-B edges 4 -> 6 of 40 and wrong state claims 3 -> 2, but exact configurations 9 -> 8 (two
  correct zero-parent counts of F1s became unresolved, one false two-parent count was withdrawn, F2_19 gained both
  true parents) and the complete view 69 -> 67 exact with 15 -> 17 extra edges (F1 siblings named as parents): the
  old penalty also held relatives down there, by accident. This is an accepted trade-off, not a passed casewise gate:
  K30's correct Tier-B claims fall 25 -> 24 while std and missing-parent gain (+1 correct claim pooled); the old
  mechanism penalised candidates for noisy withheld alleles, which is not a defensible relative discriminator.
- The predictive evidence scale rose where the scores had been miscalibrated: std 0.18 -> 1.0, missing-parent
  0.098 -> 0.19 (TroMau 0.1021 -> 0.1025, the other cases 1.0). The std and missing-parent improvements remain
  with the scale held at (approximately) its old values.
- Identical parent sets, states and tier flags under the convention reversal and two 1e-6 jitter seeds on seven
  cases (21 comparisons). Cost: none measurable.
- What it gives up: a panel-withheld allele no longer gains candidate-specific information from the parent's own reads
  (called alleles, including private ones, are unaffected). A genuinely private allele that the panel withheld, for
  example because few samples carry it, loses that evidence, and an incompatible candidate's reads no longer count
  against it there. At the traced sites the update scored worse than the background frequency; a release-aware shared
  allele model would need independent calibration.

**Linked junction transmission in the direction evidence** (`pedigree/junctions.py` `schedules`, `linked_log_cdf`;
`pedigree/scoring.py`; `pedigree/orientation.py`; `docs/methods/pedigree.md`, Finite continuous direction). The
orientation of a linked pair compared each sample's painted junction count with a binomial in which every parental
junction was inherited independently with probability 1/2: ten junctions on one parental homologue made inheriting
none look like a 1-in-1,024 event, when without a crossover it happens half the time (review control: −3.63 nats
against the true direction; now −0.66). Each painting is now traced as its minimum-switch pair of deduplicated
trajectories (the path behind its junction count, in canonical content order, so painted track order does not
matter), which places the parent's junctions on its two homologues; the transmitted homologue switches with the
meiotic selector probability between bins, and a count recurrence gives P(inherited and seen junctions ≤ the child's
count) for both directions of every eligible edge. The chromosome and pair failure mixtures are unchanged.
- Checked against independent path-enumeration references (550 and, in an independent review, 1,200 cases; ≤ 6.7e-16)
  and the review prototype (equal on every dyad of a std chromosome); the traceback's tie placement reverses under the
  convention reversal. At θ = 1/2 the selector choices at successive edges are independent, but simultaneous junctions
  on the parent's two homologues still share their edge's selector, so the former binomial is not a special case.
- On its own (linked direction against 2.15, before the allele change), in pedigree replays on the saved inputs of std,
  f3, backcross, K30, the missing-parent and N = 320 simulations and TroMau: no Tier-B or partial parent edge changed;
  on f3 two complete-view parents changed identity (F3_9: F3_85 -> F3_16; F3_22: F3_38 -> F3_26), all wrong under that
  simulation's all-unobserved truth, as before; on the missing-parent case F2_14's correct one-parent call (true parent
  F1_4) entered Tier A, its bootstrap fraction moving from 0.949 to 0.950 at the 0.95 threshold. The direction log odds
  of true parent-child pairs change individually (std F2_11 <- F1_7: 7.14 -> 9.85 nats; std median 2.39 -> 2.36;
  backcross pairs above log 49: 11 -> 17). Identical parent sets, states and tier flags under the convention reversal
  and two 1e-6 jitter seeds on all seven cases (21 comparisons). Cost (traces and dyad recurrences in the score phase):
  the pedigree stage end to end took K30 17.1 -> 15.7 s and TroMau 24.7 -> 23.5 s on the same node type, and f3 30.9 ->
  50.8 s, std 13.0 -> 16.2 s and backcross 10.8 -> 14.5 s against 2.15 runs on faster nodes (f3 traces 112 samples).

**Final family solve: early damped retry for stalled solves** (`pedigree/family.py`;
`PedigreeConfig.direction.final_progress_window = 512`, `final_pause_budgets = 4`; `docs/methods/pedigree.md`, Converged
final reciprocal solve). Some resampled fits oscillate undamped without converging, and the solve ran all 8,192 undamped
iterations before the damped restart (N = 320 simulation: two of 118 distinct bootstrap weightings, residual still 17
nats, about 50 s each, while the damped restart converges in under 300 iterations). The undamped solve now checks its
progress every 512 iterations from iteration 1,024 and pauses for the damped restart when its residual did not fall over
the last window or its rate there would need more than four budgets; if the restart fails, the undamped solve resumes
where it paused and follows exactly the iterates it would have followed, so no fit meets the residual criterion less
often. The forecast is a heuristic: slow modes that speed up later can still pause.

- Old against new policy on every distinct resampling fit of eight cases (3,846 fits, messages compared bitwise):
  identical except two fits of f3 on 8 chromosomes, which pause, converge damped and would also have converged undamped;
  both meet the residual criterion but their messages differ by up to 2.8e-7, and no pedigree table changed on any of
  the eight cases. Family solves over all fits: N = 320 139 -> 30 s; f3 on 8 chromosomes 20.4 -> 22.3 s (a paused slow
  mode pays for the retry). N = 320 pedigree stage 201 -> 148 s on the same node type.

**Pedigree memory follows the eligible edges** (`pedigree/records.py`, `pedigree/selection.py`,
`pedigree/alternatives.py`, `pedigree/m1.py`, `pedigree/calibration.py`, `pedigree/tables.py`,
`pedigree/orientation.py`, `pedigree/resampling.py`, `pedigree/panel.py`, `pedigree/selective_stage.py`,
`pedigree/stage.py`). Exact layout changes (every pedigree table byte-identical on the eight validation cases):
- the decision's genome-wide M1 profile and matched and exposed bins are [C, E] arrays on the eligible edges instead of
  three [C, N, N] arrays: the decision decodes the chromosome records one at a time and keeps copies of their
  eligible-edge entries (a decoded record's arrays can share one buffer, so a kept view would hold its dense arrays),
  and the score workers write their records instead of returning them to the driver (which held all of them as well). At
  N = 10,000 and C = 22: 52.8 GB, plus as much again for the returned records, becomes a 24 C E-byte payload, 0.53 GB
  for 10⁶ eligible edges (unchanged, 52.8 GB, without eligibility restrictions; the stage's peak also counts the arrays
  listed below);
- the junction compatibilities and the per-chromosome direction evidence are stored per oriented dyad ([C, D] and
  [C, U]) instead of [C, N, N] (at N = 10,000, C = 22 with 50 eligible parents per child: 19.8 GB -> 0.1 GB);
- bootstrap and LOCO parent counts per named edge instead of four [N, N] int64 arrays (3.2 GB -> 16 MB);
- the screen utilities are computed child by child from the compact M1 profile (no 17.6-GB [C, N, N] expansion);
- records are encoded without copying their arrays (dataclasses.asdict deep-copied every score array).
Still dense in N: each chromosome record (held one at a time), the per-fit orientation and family matrices, the panel's
screen utilities and the eligibility masks; with unrestricted eligibility the stage is quadratic in N by construction,
and N = 10,000 has not been run. A weighted sum over chromosomes on the edge axis can associate the last (E mod 4) edges
differently from the dense layout in BLAS, a one-ulp difference for at most three edges when E is not a multiple of four
(all validation cases are).

**Parents versus close relatives** (`docs/methods/pedigree.md`, Interpretation and limitations). The relationship
prototypes of this release (a second-degree relative model in which the child's homologue copies the candidate inside
IBD tracts and the population null outside, and a full-sib form) failed in their first tests because true parents
lost tens of nats on single chromosomes. That was the allele artefact above, not painting failure or relatedness.
With it fixed, every true parent of the std, missing-parent, backcross and K30 simulations has a non-negative
genome-wide contrast against the second-degree model (std 53 -> 60 of 60, missing-parent 46 -> 52 of 52, backcross
120 of 120, K30 39 -> 40 of 40). What remains is an information limit at founder resolution:
- A true parent can outscore the second-degree model by at most log 2 − Σ log(1 − θ) ≈ log 2 + Σθ per independently
  started component (about 2 nats per chromosome; the full-sib form, with two independent IBD tracts, about
  log 2 + 4Σθ), so relatives that look parent-like at founder resolution stay parent-like.
- Measured on released and complete-view edges after the fix: in f3 on 8 chromosomes (inbred, about 7 founder
  haplotypes) Tier B releases 3 full sibs as parents and 6 wrong one-parent counts among 106 claims (5 of 105 before
  the linked direction, the extra one a bootstrap fraction moving from 0.694 to 0.71 across 0.70); one of the 3 false
  edges has a negative contrast (−47.6 nats), the other two favour the parent even over the full-sib model (+6.8,
  +26.8). In K30 the F1 half-sibs and full sibs named in the complete view score +2.4 to +3.3 nats against the
  second-degree model, the same as true parents. A release abstention on a negative contrast would withhold the one
  separable false edge without losing a true parent on these simulations; it is not part of this release, and the
  reversed parent-child pairs (equally positive) show why a sign test alone is no relationship classifier.
- TroMau (no call change from the correction): 35 of its 36 released edges favour the parent over a second-degree
  relative genome-wide (median +1.61 nats per chromosome; one edge at −3.3) and none favours a full sib.

**End-to-end validation** (the release's single end-to-end run, on the candidate before a final copy-only memory repair,
through founder refit; simulations against their manifests): std 8 chromosomes, f3, backcross, K30 and TroMau all
completed (rc 0; wall 1,010, 1,031, 750, 2,202 and 6,212 s on 76-CPU nodes; TroMau 6,137 s in 2.15 on the same node
type). Every stage up to and including painting is identical to the 2.15 runs (census over 8, 3, 3, 5 and 22 contigs: no
differing founder-panel cell), and every pedigree table is byte-identical to the pedigree-only replays above. Std: Tier
B 40 of 40 exact, 60 of 60 true edges, no false edge; every released edge stays compatible with the raw genotype
likelihoods (0 of 60 excluded). f3 and backcross as in 2.16's replays (no Tier-B edge; complete view f3 16 of 112,
backcross 80 of 80). K30: Tier B 8 of 80 exact, 6 of 40 true edges, no false edge; complete view 67 of 80 with 17 extra
edges. TroMau: no call differs from 2.15 (36 released edges, 0 of 36 excluded by the raw-likelihood check). The final
memory repair (decoded records' kept arrays copied, so a view no longer holds a chromosome's dense arrays) changes no
value: pedigree replays of the final code are byte-identical on all eight cases.

## 2.15.0 (28 October 2026)

This release works on pedigree inference (stage 6). Its unobserved-parent null drew every founder state with equal
probability, so in a cross where a few founder haplotypes dominate, "no observed parent" paid for common haplotypes as
if they were rare, and a sampled relative of the child's unsampled parent could look like a parent. 2.15 draws the
unobserved parent's transmitted state from the sample's own transmitted-state frequencies, corrects the meiotic
selector to the probability of an odd number of crossovers, fits each distinct bootstrap weighting once, and corrects
the method documentation. Released results change: Tier B gains true parent edges on the std simulation, and one
TroMau two-parent call no longer meets chromosome-resampling support (it stays the complete view's call); nothing
upstream of pedigree changes.

**Sample-frequency null source** (`pedigree/forward.py` `null_frequencies`, `ChildTerms`, `_apply_null`,
`_score_path`; `pedigree/block_certificate_fast.py`; `pedigree/selective_bounds.py`, `pedigree/selective_m1.py`,
`pedigree/scoring.py`; `PedigreeConfig.scoring.null_prior_samples = 1.0`; `docs/methods/pedigree.md`, Child forward
model). An unobserved parent's slot (both slots of M0, the second of M1) now starts from, and redraws from, the
leave-one-out frequency f_cb(s) = (Σ_{j ≠ c} a_j π_jb(s) + w/S) / (Σ_{j ≠ c} a_j + w) at which the other samples with
an informative painting site transmit founder state s at bin b, with a uniform pseudo-count of w = 1 sample (a
regulariser): an empirical, persistent copying null in the spirit of the sample-allele-frequency null of parentage
likelihoods, at founder-state resolution. Per bin edge the slot keeps its state with the old kernel's persistence
weight k and otherwise redraws from f_c,b+1, so its marginal follows m_c,b+1 = k m_cb + (1 − k) f_c,b+1 (f is the
initial and redraw law, not the marginal at every bin); with uniform frequencies this is exactly the old kernel
(measured: largest difference 5.7e-14 nats of the M0/M1 screen with the frequencies forced uniform). A saturated
kernel whose persistence weight rounds below zero is set to zero persistence; a materially negative weight is an
error. The production M1 certificate's null step uses the same law (every term stays nonnegative); the private
relaxation bound only needs the null's unit row sums. The frequencies pool every informative sample (descendants and
large sibships included): they describe the sampled population, and leaving the child out removes its own row, not
its influence on the shared paintings. Cost: one float64 [N, B, S] array per component (8 N B S bytes), O(N B S).
- Validation (pedigree replays of the 2.14 end-to-end runs' saved paintings and evidence, all other stages fixed;
  simulation truth; parents compared as sets): std Tier B 38 -> 39 exact configurations of 40, 57 -> 59 of 60 true
  edges, 0 extra; f3 (every sample truly without observed parents) complete view 13 -> 16 exact, false edges
  136 -> 127, Tier B unchanged (5 certified M0 states); backcross unchanged; K30 Tier B and complete view unchanged
  in aggregate, but its supported partial edges fall from 10 to 8 true (F2_2's two parents and one of F2_3's lost,
  one of F2_17's gained; no false partial edge) and 3 samples' Tier-B state flags change. TroMau: complete view
  identical; Tier B releases 2021cicX14537327's two parents no longer (its genetic M2-over-M1 contrast falls from 61.0
  to 17.3 nats before the predictive scale, the marginal state margin from 5.0 to 0.38 nats, bootstrap support
  0.87 -> 0.55; the call stays the complete view's), and 3 samples' Tier A flags change. Over TroMau's released
  calls the state margins fall by a median 2.6 nats (one parent) and 3.4 nats (two parents) genome-wide: a common
  founder state is cheaper for an unsampled parent, which weakens relatives and true parents alike; the change is not
  monotone in parentage accuracy.
- A new missing-parent simulation (cohorts 10/20/30, 75% of individuals sequenced at random, 8 chromosomes; 45
  samples: 8 with no, 22 with one and 15 with two observed parents): Tier B 43 -> 44 exact of 45 (the last M0 child
  recovered; one-parent 22/22 and two-parent 14/15 unchanged), 50 of 52 true edges, no false edge or wrong state
  claim; complete view 45/45 in both. A larger simulation (N = 320, cohorts 20/100/200, 5 chromosomes): 320/320
  exact, 600/600 edges in both.
- The pseudo-count w = 1 is a fixed choice, not tuned. Sensitivity (std, f3, K30, TroMau, missing-parent; backcross
  not run): w = 0.1 gives identical calls everywhere; w = 10 lowers std's Tier B from 39 to 37 exact configurations
  and from 59 to 55 true edges (partial 59 -> 57; complete view and false edges unchanged) and moves one K30 Tier-B
  state flag and one missing-parent Tier-A flag. A stronger pull towards uniform frequencies loses true std parents.
- Every released edge of std and TroMau stays Mendelian-compatible on the raw genotype likelihoods (full-marker
  exclusion log E at most -7.19 and -3.74, against release thresholds of 8.99 and 10.05).

**Meiotic selector: odd crossover parity** (`pedigree/prepare.py`; `model/genetic_map.py`; `docs/genetic_maps.md`).
Which parental homologue a child receives changes between two bin centres when an odd number of crossovers falls
between them, θ = (1 − e^(−2d))/2 (Haldane's recombination fraction of one meiosis, `haldane_fraction`), clipped to
[1e-15, 0.5]. 2.14 used 1 − e^(−d), the probability of at least one crossover, which treats an even number as a
switch (the projection applies θ as a homologue flip). The largest relative difference is about 2.3% of θ on TroMau's
bins and 0.6% on the simulations; no released or complete-view call changed on the five validation cases (TroMau
included). The now-unused `poisson_switch_stay_terms` helper is removed.

**Bootstrap fits once per distinct weighting** (`pedigree/resampling.py`). A resampled fit is a deterministic function
of its chromosome multiplicities, so each distinct multiplicity vector of the 1,000 seeded draws is fitted once and its
selections, failures and fits are counted as many times as it was drawn (LOCO stays separate). With few chromosomes
most draws repeat (10 distinct vectors at C = 3, 118 at C = 5, 840 at C = 8, all 1,000 at C = 22). Every pedigree
table is byte-identical to fitting each draw (five cases; only the calibration report's timing fields differ). End
to end, K30's pedigree stage took 17.1 instead of 27.9 s; with 22 chromosomes (TroMau) every draw is distinct.

**Method documentation** (`docs/methods/pedigree.md`). The cost section no longer claims a 190-row M2 panel per child
or O(N²) decision rows: the retained-parent panel has no cap (measured: f3 43,052 rows, at most 2,415 per child), the
decision has N + E + T rows, and the screen's C × N × N profile and the downstream arrays are dense in N (17.6 GB per
float64 array at C = 22, N = 10,000). The tie tolerance states its 2^-15-nat floor; the unordered parent slot order
states its name fallback for identical observed-site counts; two limitations are new (junctions on one homologue are
transmitted together, which the binomial direction comparison ignores; a sampled relative of an unsampled parent can
outscore the null, see below).

**Measured, not changed.**
- Stability: 1e-6 genotype-likelihood jitter (two seeds) and the convention reversal of the pedigree stage
  (upstream records fixed) leave every released, partial and complete-view parent set and every state and tier
  flag unchanged on the five cases, for 2.14 and 2.15 (the unordered Parent1/Parent2 label flips under the reversal,
  as documented).
- Parent versus related non-parent. A prototype related-but-not-parent slot (the child's homologue copies a candidate
  inside IBD tracts that switch at the meiotic rate, the population null outside) scores true parents above it by
  about 2.2 nats per chromosome (median; std and backcross; 0.6 on K30), while in the inbred f3 design full sibs fit
  it better than as parents (by about 2.2 nats per chromosome). Outbred designs already reject relatives through
  allele mismatches (M1 − M0 medians of −600 to −3,000 nats per chromosome). Mixing it into the decision's
  unobserved-parent explanations was tried on the saved scores and these tested extensions were rejected: into M0
  alone (per chromosome or genome-wide, mixture weights 0.1-0.9) it improves f3's complete view (16 -> 35-59 exact)
  and, genome-wide, K30's Tier B (4 -> 18 true edges), but loses up to 7 true std Tier-B edges and creates 7-18 wrong
  Tier-B state claims in the missing-parent design; into M1's unobserved slot as well, it calls true two-parent
  children one-parent (std: 9 wrong claims and a false edge). A relative resembles a partially IBD parent, and these
  mixtures let it compete with the parent hypothesis for the same candidate; a joint relationship classification
  (parent, full sib, half-sib or grandparent, unrelated) is the proposed next design.
- Tier-B state claims (all released parent counts, including those whose identity is unresolved) are not error-free
  where relatives are numerous and founders hard to tell apart: K30 has 3 wrong of 28 (4 of 29 in 2.14), std 1 of 40
  (a two-parent child claimed as one-parent, its released edge right). No released Tier-B edge is false in any
  validation case.
- Resampling at N = 320 (5 chromosomes) takes about 50 of the stage's 120 s and scales poorly with workers: 4 to 112
  workers take 53 to 60 s for 118 distinct fits of about 1 s each, one worker 116 s; process start-up is a suspected
  contributor (not separately timed).
- f3 and K30 are difficult low-diversity designs, poorly separated by the present model; these tests do not separate
  limited information from model inadequacy. f3 has about 7 founder haplotypes per chromosome and sibling matings
  (kinship 0.22-0.50 between "unrelated" candidates; the full-marker exclusion leaves a compatible relative for most
  samples on 3 chromosomes); with K30's low-divergence templates a true parent scores about 2 nats per chromosome
  over M0, no more than a half-sib.

**Known limitations.** The null does not model a sampled relative of the unsampled parent (above). The direction
term's binomial transmission of junctions ignores their linkage on a homologue (bounded at log(2/ε_c − 1) nats per
chromosome). Pedigree memory and decision arrays are dense in N; N = 10,000 needs a sparse representation. The
ancestry-path factor's top-16 cutoff has an exact-equality boundary (no released call moved under the jitter).

**End-to-end validation** (the release candidate, from scratch through founder refit, family phase skipped as in every
validation since 2.7; inputs, configurations and seeds of the 2.13 final runs). On all five cases (std, f3, backcross,
K30, TroMau) every stage up to and including painting is identical to 2.14 (0 differing founder-panel cells), founder
refit is identical (it does not read the pedigree), and the pedigree tables are byte-identical to the stage replays
above. Wall times 694, 714, 532, 2,230 and 6,150 s (2.14: 701, 747, 548, 2,317, 6,298; nodes of the same types); the
pedigree stage took 13.0, 30.9, 10.8, 17.1 and 24.7 s (2.14: 12.7, 32.7, 12.1, 27.9, 29.9). The candidate that ran
end to end differs from the release only by the saturated-kernel repair (byte-identical stage replays on all five
cases) and text.

## 2.14.0 (27 October 2026)

This release works on painting (stage 5). Its switch prior was an imposed switch-suppression prior: every track change
cost about 109 nats (theta exp(-100)), which kept switch precision high but made the within-model posteriors near 0 or
1 (on the std simulation, among pairs with qV >= 1 - 1e-6 where the panel held the truth, 0.063% were wrong against a
claimed 2e-10). 2.14 replaces it with a copying model whose parameters are estimated from the data, an explicit
panel-error state and a design-effect adjustment of the emissions, and gates the release on the probability of
faithful copying. It also makes the released track order and label continuity's tie decisions independent of the
traceback convention, fixes a switch-interval predicate and a label-continuity switch count, and reports the
founder-identity evidence of label continuity's moves. Released results change: the painting (paths, posteriors,
abstention), label continuity's moves on some components, pedigree's Tier B release on a few samples, and the founder
refit through the painting.

**A copying model in place of the switch penalty** (`painting/transitions.py`, new `painting/fidelity.py`,
`painting/stage.py`; `docs/methods/painting.md`, Copying model). Each track copies a label (a trajectory class or
BACKGROUND) with a fidelity (faithful, or in a panel-error run; anchored classes only).
- Labels change along the genetic map as a jump process at rate rho per Morgan, each jump to one of the other S - 1
  labels uniformly, independently on the two tracks: theta = (S - 1)/S (1 - exp(-rho d S/(S - 1))) per bin edge of d
  Morgans. The pair transition is the product of the two tracks' (2.13 coupled a change of both tracks with an
  exponent of 1.5).
- A label that stays enters a panel-error run with probability a per edge and leaves it with probability b; a label
  that moves, and a component's first bin, draw the fidelity from the chain's stationary law; an error track emits as
  BACKGROUND (its row is locally wrong for the sample, or the sample's haplotype is locally absent), so a locally wrong
  row contributes log(1/a) at the run's entry and log(1/b) at its exit (besides its duration and label-stay factors)
  and keeps the founder label, instead of two ancestry changes. Painted switches are label changes only.
- Each sample's bin log emission is divided by 1 + (n - 1) r, n the sample's informative sites in the bin and r =
  `PaintingConfig.emission_site_correlation = 0.1` (`painting.hmm.design_effect`): a tuned, equicorrelation-inspired
  magnitude adjustment of the composite log likelihood (Kish's design effect for n equal-variance terms with common
  correlation r; under that working assumption a bin of 100 informative sites counts as about 9 variance-equivalent
  sites and one of 8 as about 4.7, and the weight tends to 1 as n -> 1). r is a tuned constant, not an estimated
  correlation or dependence correction.
- (rho, a, b) are estimated per component from all its eligible samples by EM: 8 updates from fixed starts (rho 2 per
  Morgan, a 1e-3, b 0.25), each an exact forward-backward pass over the chain followed by the M-step of the expected
  complete-data log-likelihood (rho by a scan and golden-section search; a and b by bounded quasi-Newton, including
  the stationary fidelity draws at the first bin and after label moves), neither update accepted when it would lower
  its part of the objective; rounded to 2^-6 in log space. A truncated, deterministic estimator, not a convergence
  test (for the constant-temperature predecessor, 4, 8 and 16 updates gave the same released decisions on std and
  backcross). The model is estimated before label continuity and fixed through it; per component it is published in
  the new `painting_model.tsv` and summarised in the log.
- Release: the gate is qF (qAF at stage 2), the model's posterior mass that both tracks copy the Viterbi labels' rows
  faithfully, at the unchanged tau = 0.9; qV and qA remain as diagnostics (qA >= qV >= qF).
- Consumers. Label continuity screens candidates with the model's label-level process (the estimated rate, tempered
  emissions, no error state: the error mixture of a bin is not a sum over the parts of a bin a link splits; each part
  is divided by its whole bin's design effect) and decides every candidate on the painted switches of the full model;
  a candidate that would change which of its classes are called anywhere (their error states) is not accepted.
  Pedigree's source chain is a reduced model: the label-level table and the tempered emissions (packed, or recomputed
  and divided by the design effect), without the panel-error state; a called founder allele is still transmitted
  unchanged. Records: painting schema 3 (the transition record carries rho, a, b and r).
  `PaintingConfig.switch_penalty_per_snp` and `double_recomb_factor` are removed; `emission_site_correlation`,
  `initial_switch_rate`, `initial_error_entry`, `initial_error_exit`, `estimation_iterations` and `estimate_grid` are
  added (internal constants). The 2.13 Viterbi and posterior kernels of `painting/hmm.py` are removed.
- Validation against simulation truth on the 2.13 final panels (painted switch events matched one to one, maximum
  cardinality, with true ancestry changes within 3 bins; a decoded pair counts as right when its rows carry the true
  haplotypes' alleles in the bin up to max(1, 2%) mismatches; mixed bins excluded), against 2.14's other fixes on the
  2.13 prior, label continuity on: std matched events 2,166 -> 2,162 of 2,311 true with painted events 2,389 -> 2,334
  (precision 0.907 -> 0.926, recall 0.937 -> 0.936); f3 886 of 1,007 matched in both, painted 961 -> 959 (precision
  0.922 -> 0.924); backcross matched 1,786 -> 1,784 of 1,809 with unmatched painted events 9 -> 12 (precision 0.995 ->
  0.993, recall 0.987 -> 0.986); K30 equal; wrong pairs where the panel held the truth std 609 -> 546, backcross 79 ->
  67, f3 and K30 equal; pairs whose rows' whole-component founder identity mismatches the truth std 43,958 -> 43,435,
  backcross 102 -> 78. A trade-off, not an improvement on every count: std loses 4 matched events, backcross 2. Most
  wrong pairs remain where no row of the panel carries the truth (std 8,769 of 9,403), which a named-panel label
  cannot correctly represent.
- Calibration on represented truth (std, 8 contigs; observed local-allele error against mean predicted 1 - qF, by
  bucket; qF's faithful latent event differs from the scored local-allele event): qF < 0.5 1.4e-2 against 0.64,
  [0.5, 0.9) 8.0e-3 against 0.26, [0.9, 0.99) 4.0e-3 against 4.0e-2, [0.99, 0.999) 2.2e-3 against 3.4e-3,
  [0.999, 1 - 1e-6) 9.5e-4 against 3.4e-5, >= 1 - 1e-6 7.6e-5 (38 of 497,327) against 3.8e-7; 2.13's qV had 4e-3 to
  5.5e-3 in every bucket from 0.5 to 1 - 1e-6 and 6.3e-4 against 2e-10 at the top. On these diagnostics qF is closer
  to its observed errors than 2.13's qV, not calibrated: lower in the low buckets (partly the event mismatch), about
  equal in [0.99, 0.999), over-confident above 0.999 (about 30 and 200 times). The same panels served to tune and to
  measure the model.
- Costs: abstention rises (std 234 -> 3,112 sample-bins of ~945k, backcross 104 -> 1,087, f3 12 -> 373, K30 2 -> 32);
  painting takes longer (8 estimation passes plus the decode and label continuity's chain re-decodes, each
  O(N B S'^2) with S' <= 2S - 1 track states): a synthetic component with N = 300 samples, 2,000 bins and 200 link
  points (112 threads) at K = 10 / 20 / 30 takes 14 / 51 / 110 s to estimate, 11 / 38 / 82 s for label continuity and
  3 / 10 / 21 s to decode and release (2.13: 6 / 19 / 40 s in all), peak memory 10 / 26 / 50 GiB (2.13: 6 / 11 / 20
  GiB); label continuity's whole-search memory at K = 30, N = 300, 10,000 bins and 2,000 link points is about 242 GB
  (2.13's model: 79 GB), batched under a smaller grant.
- Without the emission adjustment the estimated rate and the error chain lose precision (std 0.772, f3 0.692, backcross
  0.800): where a row is locally wrong the path escapes it by changing label. A fixed temperature (every bin divided by
  T = 12) came close to 2.13's precision on the simulations (below it on backcross) but lost a sparsely read true short
  tract in a frozen control (no switch, wrong labels), which the design effect recovers exactly (min qF 0.985); on the
  simulations' full bins the two behave alike. Design-effect sweep without label continuity, r = 0.1 / 0.125 / 0.15: std
  precision 0.925 / 0.925 / 0.922 at recall 0.936 / 0.934 / 0.932; f3 precision 0.924 / 0.928 / 0.928 (recall 0.880);
  backcross precision 0.993 / 0.993 / 0.992 at recall 0.986 / 0.984 / 0.981; K30 unchanged. Without the panel-error
  state, precision falls where the panel has error runs (std chr1 0.899 -> 0.855). Rejected on the way: an independent
  per-bin panel-error mixture (std precision 0.696), an allele-level miscopy rate (lower still) and an error state that
  copies its row with flips (0.60-0.68).
- Downstream (copies of the 2.13 final runs repainted and continued to founder refit): pedigree against truth std
  exact parent configurations 37 -> 38 and true edges found 55 -> 57 of 60 (none extra), f3, backcross and K30
  unchanged; TroMau parent assignments and states unchanged for all 116 samples, Tier B release changed for 3 (two
  two-parent calls released, one zero-parent call withheld). Founder refit against truth (wrong or missing alleles)
  rises slightly: std 461,079 -> 461,529, backcross 924 -> 978, f3 223,984 -> 224,001, K30 55,717 -> 55,725 (the refit
  reads fewer released labels). TroMau painted panels against the G0 pseudo-reference (dependent reference, not
  truth): wrong alleles at W0 -14.6% against 2.10 (2.14's other fixes alone: -12.9%), at W100 equal.
- Stability: on the five validation cases (std, f3, backcross, K30, TroMau) the released painting is identical under
  the convention reversal and under a 1e-6 jitter of the genotype likelihoods (two seeds). The decoder chooses a label
  change from the first source within `VITERBI_TIE` of the best; the reversed convention decodes the same model
  backwards (transposed transition, the initial law at its last step); a dense brute-force reference checks the
  kernels and the EM's monotone likelihood.

**Released track order from the unordered candidate sets** (`painting/hmm.py` `canonical_track_order`;
`painting/fidelity.py` `decode_candidates`). The two tracks' order (H1/H2, a gauge) is fixed from the bins' combined
unordered candidate sets: from the last bin (smaller candidate mask on track 0) leftwards, each bin takes the
orientation with fewer track changes towards the next bin, then fewer allele-visible changes, then the smaller mask on
track 0. It no longer depends on which traceback convention ran: the default and the reversed convention release
identical ordered products on all five validation cases (before: the tracks of 52-97% of sample-bins exchanged, the
unordered product equal). Measured alone against 2.13 (before the model change below), the unordered product was
unchanged and the order differed in up to 5.1% of sample-bins (TroMau); pedigree's tables and founder refit's summary
were byte-identical under it (copies of the 2.13 final runs repainted and continued to founder refit).

**Label continuity's ties** (`painting/label_continuity.py`).
- An accepted transposition is placed at the leftmost link point of a run where the two rows carry identical
  founder-axis data (the same per-site panel data and painting at either point), and such equivalent placements are
  checked once; the convention reversal moved it to the rightmost before (TroMau chr3, chr6, chr14; identical calls).
- A stated policy for competing repairs: when another acceptable candidate of the accepted one's tie class of gains
  would relabel differently (another row pair, or a non-equivalent link), neither is applied in that iteration and the
  event is counted (`painting.relabel_ambiguous`). Before, the first in index order was taken, so the input row order or
  the convention reversal chose between founder panels (a review fixture: 450 cells apart after the best row matching;
  now no move under every row order, convention and sample order). No such tie occurred on the validation panels.
- Painted switches are counted as the painting reports them, the fewer of the two conventions' paths of the new decoder
  (before: the run's convention alone), and every eligible sample is re-decoded for each distinct candidate. Before,
  samples whose best unaffected state strictly beat every affected one were skipped; a relabelling can change which of
  several exactly co-optimal paths a traceback convention selects, and so the counted switches, so that shortcut was not
  a certificate (a review's finite HMM counterexample; none in 1,500 genotype-emission fixtures). A check keeps only the
  per-sample count changes (a review fixture with 999 link points: 801 checks and 103 MB of retained candidate maps
  before, now 1 check and 0.3 MB). No accepted move changed on the validation panels.
- Memory: the search is granted as much as the measured headroom admits, without the former ceiling (the larger of 32
  GiB and a quarter of the run's limit), so a large-K search batches (about 3x slower, identical results) only under
  real pressure (K = 30, N = 300, 10,000 bins, 2,000 link points: about 79 GB at once under 2.13's model, about 242 GB
  under 2.14's).

**Switch intervals use the emission's equivalence** (`painting/hmm.py` `_distinguishable`, `switch_interval_masks`). Two
classes are interchangeable for a track in a bin only when, paired with the sample's other track, they have the same
genotype law at every observed site: where both are uncalled and the other track is a named class equal to one of them,
that pair shares one latent allele and the other does not. Before, every equal call counted, so a switch that the reads
located could be released as an interval (a review fixture: two bins withheld as SWITCH_INTERVAL against a 108- and
489-nat preference; now named). No released bin changed on the validation panels.

**Founder identity across label-continuity moves (diagnostic)** (`painting/release.py` `RelabelMove`, new
`label_continuity.tsv`). A label exchange of two founders' rows makes the carriers of either founder switch at the link
in both directions; one shared recombinant lineage switches one way only. Each accepted move records how many winners
switch each way and is `identified` when both are positive; a one-way move is kept and reported as identity-ambiguous
(log: `(k identity-ambiguous)`). No consumer reads the flag, so it changes no released label: it does not solve founder
identity, it shows where the evidence for it is one-sided.

**Viterbi tie band per candidate.** The decoder judges each candidate predecessor on its own score against the band of
the maximum (2.13 compared the row and column one-change candidates with each other first, which could admit a pointer
up to twice `VITERBI_TIE` below the best). The dynamic-programming scores are exact maxima; with two stages per bin the
traced path lies within (2B - 1) x `VITERBI_TIE` of them.

**Measured, not changed.**
- Label continuity is kept. Over the saved 2.12 and 2.13 simulation runs its 10 accepted moves (3 to 6 winners, no
  loser) improved founder identity in some cases (std chr10 -24.8k, chr18 -0.9k, chr20 -16k, K30 m5 -0.5k to -0.8k
  wrong-or-missing alleles) and worsened it in others (std chr20 +16.9k, chr1 +1.8k), with no truth-free feature found
  to separate them. Under the 2.14 model TroMau has 8 moves (2.13 prior: 11; 6 in common).
- Speed: painting was about 0.3% of a TroMau run under 2.13 (114 s of 42,370 lane-seconds).

**Correction to 2.13.** 2.13's known limitation said a 1e-6 jitter decides an exact assembly tie the other way "by
breaking the equality". The tested 1e-12 jitter (seed 1), which breaks the equalities of the genotype likelihoods
themselves, leaves the std run unchanged at every stage (chr1 included), so those equalities alone do not decide it: the
1e-6 jitter moves a convention-decided comparison the same way in every seed (a tolerance between the two scales, or a
systematic effect of the jitter). The decision is not yet located.

**Known limitations.** The design effect's r is a tuned constant chosen on simulations (the same 2.13 panels served
to choose and to measure it), not a measured correlation; real panels may err differently, and qF is a within-model
probability, over-confident above 0.999. Pedigree's source chain and label continuity's screen are reduced models
without the panel-error state; pedigree transmits a called founder allele unchanged even where painting puts the row
in an error run. Label continuity remains a heuristic for founder identity (above); its identity flag is a
diagnostic. Bins repartition a component's selected sites globally, so one inserted or removed site can move every bin
edge of the component (a local binning is a model change left for a later release). Label continuity's candidate
checks are at worst quartic in K per iteration (every screened candidate re-decoded); the validation runs checked at
most 8 distinct candidates per component.

**End-to-end validation** (the release candidate, from scratch through founder refit, family phase skipped as in every
validation since 2.7; inputs, configurations and seeds of the 2.13 final runs). On all five cases (std, f3, backcross,
K30, TroMau) every stage before painting is identical to 2.13's (discovery, local selection, assembly and founder
refinement; zero differing founder cells), and the painted panels are identical on the four simulations. The
simulations' painting, pedigree and founder-refit results equal the targeted replays above (founder refit
wrong-or-missing alleles std 461,529, f3 224,001, backcross 978, K30 55,725; pedigree std 38 exact parent
configurations and 57 of 60 true edges, none extra). TroMau: label continuity makes 8 moves (2.14's other fixes on the
2.13 prior: 11), so the painted panels differ in 39,749 founder cells; against the G0 pseudo-reference (dependent
reference, not truth) wrong alleles at W0 fall 14.6% against 2.10 (2.14's other fixes alone: 12.9%) and at W100 5.9%
(equal); parent assignments and states are unchanged for all 116 samples and the Tier B release changes for 3 (94 ->
95 released). Wall time: TroMau 6,298 s (2.14's other fixes alone: 6,399 s), K30 2,317 s (2,257 s) on full 76-CPU
nodes.

## 2.13.0 (27 October 2026)

This release reduces the demonstrated sensitivity of stages 1 to 5 to a 1e-6 relative change of the genotype
likelihoods. Releases 2.1 to 2.12 placed every tie quantum against 1e-9 noise. Under a 1e-6 jitter, compared values
left their tie classes: discovery changed 19 to 37 blocks per std or TroMau run and 144 of K30's 200 blocks, local
selection's context feedback grew a few such blocks into tens of changed blocks, and assembly carried them into
different founder panels; founder refinement's polish also chose among near-equal candidates by the last bits of their
gains. Under the same jitter, discovery and local selection now agree in every measured seed and contig (Validation).
Released results change.

**Decision tolerances above the noise floor** (`maths/ties.py`, `assembly/tie_rules.py`, `discovery/landmarks.py`,
`local_selection/fit.py`, `local_selection/search.py`, `local_selection/context_refit.py`, `local_selection/config.py`,
`assembly/founder_alleles/site_posterior.py`, `assembly/founder_refinement/acceptance.py`,
`assembly/founder_refinement/proposals.py`, `painting/hmm.py`). Owner decision: no decision tolerance below the noise
floor. `maths.ties.GL_NOISE = 1e-6`; a quantum found below that floor becomes `noise_quantum(motion)`, the smallest
power of two at least 16 times its compared contrast's motion under that noise, measured on frozen contrasts of 1e-6
jitter runs where available, otherwise the 2.1 census times 10^3 (a relative factor without a recorded motion: its
2.1 value times 2^10). The motion is a typical measured or extrapolated one, not a worst-case certificate.
- Discovery: `nll` at least max(2^-10, 2^-20 M), `h_step` 2^-12, `improvement` 2^-10, `sample` at least
  max(2^-13, 2^-26 M), `cell` 2^-14, `probability` 2^-15, also the landmark gap and the residual and partner bands
  (2^-17 to 2^-23 floors before). Replays of every discovery block that differed between a 2.12 run and its jittered
  twin (TroMau 126, std 63, K30 144 blocks) now agree under both evidences; replays of every block of TroMau chr13
  (380) and K30 m4 (40) and every third block of std chr10 (501) under one seeded 1e-5 jitter change no block (a
  probe, not a worst-case guarantee). Against 2.12 on the same contigs, discovery's wrong-or-missing alleles change
  by +0.07 to +0.5% and its called-allele errors mostly fall (its release counts slightly fewer weak supporters).
- Local selection: its decision tolerance stays 2^-10 (a search round's gain moves by about 5e-5 nats at the 99.9th
  percentile on jitter runs); the
  consensus band is 2^-20 (2^-31), the flip band 2^-10 relative (2^-20), the release's favourable margin 2^-14
  (2^-23) and unknown-count and background-count bands 2^-15 (2^-20).
- Assembly: founder refinement's quantum is max(2^-9, 2^-26 |F|) (max(2^-14, 2^-35 |F|)) and its acceptance
  threshold 2^-9 nats (1e-4), from the measured motion of refinement contrasts (p99 9e-5 nats); the window
  programmes' canonical floor is 2^-14 (1e-7); the site posterior's half band 2^-20 (2^-26) and its improvement
  2^-10 (2^-17). The join quanta are unchanged: frozen join contrasts move 32 to 63 times less than them.
- Painting: the Viterbi tie is 2^-15 nats (2^-20).
Not changed: convergence tolerances of fits, float-resolution margins of exact bounds, stated statistical thresholds;
not yet re-measured: founder refinement's guard tolerance and local selection's flip-acceptance tolerance (both
1e-6); family phase's chain ties and gain quantum (outside the validated stages). Pedigree's tie classes read the
rebased `maths.ties` quanta.

**Founder refinement's polish decides near-ties by content** (`assembly/founder_refinement/beam.py`,
`assembly/founder_refinement/search.py`). The escape beam ranked a unit's children on a fixed grid of their total
bounds, which a 1e-6 perturbation shifts in common by several quanta; it now quantises bounds relative to the unit's
best. Long repair and bridges chose the best variant, precursor, exchange and bridge by exact value; each now takes the
content-smallest candidate within the refinement quantum of the best. std chr10, which differs under the jitter in
all four of 2.12's seeds (603 to 1,413 founder cells), differs in one of four (982.5 cells, at assembly level 4).

**Validation.** The single end-to-end run of this release, to painting, on 2.12's cases and contigs. Simulations
against truth (WOM: truth-to-panel wrong plus missing alleles under one-to-one matching):
- std, 8 contigs: WOM 465,652 -> 465,987 (+0.07%; chr10 and chr20 differ), nearest-row called errors +0.10%,
  missing alleles 32,043 -> 32,035, label switches 718 -> 716.
- f3: WOM +0.01% (chr9 only). backcross: identical.
- K30: WOM 62,529 -> 55,965 (-10.5%), but one-to-one called errors 3,716 -> 4,880, nearest-row called errors +4.0%,
  missing alleles +4.8% and label switches 24 -> 27. A 1e-6 jitter alone moved 2.12's K30 WOM by -12.1% and its
  switches from 24 to 33, so this difference is of the size of 2.12's noise-chosen states, not a measured model effect.
- TroMau, against the G0 v1.4 pseudo-reference (derived from earlier reconstructions of the same data; consistency,
  not accuracy), window-matched (W100) disagreement against 2.10: wrong 316,080 -> 297,390 (limit +22,168), wrong or
  missing 482,994 -> 464,852 (limit +22,093); against 2.12 wrong -4.7% and wrong or missing -3.1%, mostly chr7, chr9
  and chr14 (-3.7k, -5.4k, -5.2k), with chr13 +0.5k; on 13 of 22 contigs W100 wrong is within 10 alleles of 2.12.
- Run time within 5% of 2.12 on every case (TroMau -0.6%).

Stability, founder cells that differ at painting between a run and its 1e-6-jittered twin (2.12 -> 2.13):
- std, 8 contigs, four seeds: 1,268 / 5,700 / 2,042 / 1,717.5 -> 228.5 / 167.5 / 1,150 / 228.5.
- TroMau, 22 contigs, four seeds: 48,195 / 94,918.5 / 59,829.5 / 27,790 -> 31,604 / 32,458 / 32,705 / 32,305, of
  which 31,601 on chr7 in every seed (the rest 3 / 857 / 1,104 / 704). The jittered runs' W100 wrong spans
  297,379-297,394 (2.12's twins 310,687-312,108).
- K30: 11,950.5 -> 32. f3 and backcross: 0 in both.
- Discovery and local selection agree in every seed and contig; the remaining differences arise at assembly levels 2
  to 4. TroMau chr7 (31,601 cells) and std chr1 (167.5) differ identically in every seed while the jittered runs agree
  exactly with one another, so there the unjittered run is the one that differs (Known limitations).
- Convention reversal (the opposite direction on every exact tie): std 249 cells (2.12: 249, on the same contigs);
  TroMau 52,173 (2.12: 23,627, none on chr7), of which 31,601 on chr7, where the reversed run's final panels equal
  every jittered run's; the other 21 contigs 20,572.

**Known limitations.** Exact ties in assembly are still decided by convention. On TroMau chr7 the
convention-reversed run and every jittered run reach identical final founder panels (31,601 cells from the
unjittered run's, diverging from assembly level 1 or 2), and on std chr1 they share the same level-3 change (83
cells): an exact tie there is decided by convention, and a jitter, by breaking the equality, decides it the other way
in every seed. Equal read counts give exactly equal genotype likelihoods in real data, so such ties are expected; the
decision is not yet located. 2.12's limitations of founder refinement's objective and bounded search, of painting's
identity moves and of discovery's path dependence under exact ties are unchanged.

## 2.12.0 (27 October 2026)

This release removes the read-model calibration's site subsampling and the demonstrated discontinuities of local
selection's context code, adds trajectory retention and a selection-aware local improvement pass to local selection's
fits, and lets local selection's release credit alleles that the context refit witnessed. A calibration redraw, or a
1e-6 change of the genotype likelihoods, could tip near-threshold local-selection decisions that assembly and founder
refinement then amplified into different local optima. On the std eight-contig redraw (targeted replays below), none
of the replayed selections trails the tested candidate bank by more than one nat under its own evidence and context,
and 49 of 191 pairs select identical panels; other search differences remain, mainly on TroMau, and founder refinement
still lands in different optima (Known limitations). Released results change.

**Read-model calibration on every kept site, with dependence-aware selection** (`evidence/read_model/calibrate.py`,
`evidence/read_model/fit.py`, `evidence/config.py`, `evidence/stage.py`, `evidence/reports.py`). Releases 2.3 to 2.11
fitted the read model on about one kept site in eight and chose among the shared binomial, sample binomial and sample
joint candidates by the summed held-out score of one fold within a one-nat band. Redrawing that sample moved the
contigs' mean fitted error logits by about 0.1 (standard deviation over 22 TroMau contigs) and changed the selected
model on 7 of them. Calibration now:
- builds the histograms from every kept site (cells subject to the observed-cell and maximum-depth admission);
- fits on folds 0/1 and scores folds 2/3;
- accepts each more complex candidate over the currently accepted one only when its held-out gain is at least
  `ReadModelConfig.selection_z = 2.0` standard errors.

The standard error comes from the held-out 2-Mb stretches that hold admitted cells. Sites are not independent: on six
TroMau contigs the standard error of the held-out gain grew 4- to 16-fold from single sites to 2-Mb windows (the
windows set the scale of the error estimate; they neither bound it nor identify a dependence range). With fewer than
`selection_minimum_blocks = 4` such stretches the shared binomial is kept. Each comparison records its gain, error, z,
stretches and leave-one-stretch-out flips; selections within 0.5 of the threshold are counted in the ties report. The
empirical-Bayes shrinkage weight depends on the amount of data, and a per-sample error rate near 1e-5 is weakly
identified at low depth (docs/methods/evidence.md). This is a reasoned estimator change, not a demonstrated accuracy
improvement.

**Local selection: the context code's demonstrated discontinuities removed** (`local_selection/fit.py`,
`objective.py`, `context_refit.py`, `search.py`, `rounds.py`). In a context round the assembled context's founders
code the panel rows they claim. A perturbation could change a block's selected panel by tens to hundreds of nats in
four demonstrated ways:
- **Trajectory-best selection.** The fixed-K GEM climbs the context-free objective while rounds select fits by the
  context-coded objective, so a fit could pass through a panel the context code prefers and then leave it. A context
  round's fit now returns the best state under the selection objective along its trajectory when it beats the
  endpoint by more than the decision tolerance, with its frequencies refitted.
- **All-edge, all-cardinality context code.** Each founder used to claim only one of its nearest rows (maximum
  cardinality first) at the shell code log C(n, d), so one changed call could move a claim between nearest rows
  and cost a whole free row. The code is now the shortest description over every matching of founders to rows, of
  every size, at the monotone edit-ball code log sum_{j<=d} C(n, j), computed exactly by successive shortest
  augmenting paths in O(min(F, K)(F + K)^2), with one canonical matching (independent of row and founder order).
  For fixed claimers and fixed called masks a changed call moves the code by at most log(n + 1).
- **One claimer per assembled founder.** The context refit merged founders whose local rows were equal, so a
  collision of two founders' rows removed a claimer: 200 log 2 nats for one call in the observed case. The
  proposal panel and the refit still use the distinct rows, but the code keeps one claimer per assembled founder
  (`ContextProposal.multiplicity`).
- **Context snaps.** The likelihood-only GEM left data-uninformative alleles wherever its trajectory put them. A
  context round's fit now ends with one-bit steps judged by the selection objective: proposals are the called-site
  disagreements of the matched pairs of one canonical shortest description, ranked by a valid lower bound (exact
  one-bit likelihood gain plus the pair's edit-code saving). A step needs the largest bound above the decision
  tolerance; bounds within the tolerance of it are one tie class, decided by content and counted (`ls.snap.tie`);
  the step is accepted only if the recomputed selection value rises. Each step costs O(K^3 + K L) plus
  O(K L log K) per tie-class member. This is a local improvement pass, not a complete search: a bit outside the
  matched disagreements, or one that would create a claim, is not proposed.

The code charges a claimed row only on its founder's called sites (a context-reuse penalty, stated in the methods,
not a complete code). One whole-row transition is therefore retained: when a founder's last informative call
disappears it stops claiming (a single perfectly matching call in a 200-site block moves the regulariser by 138.6
nats). Two completions were tested against simulation truth and left out, each a tradeoff rather than a gain: coding
uncalled sites (K30: summed Hamming distance 859 -> 844 but exactly recovered true rows 299 -> 296) and a radius code
(std: summed Hamming distance 908 -> 1,097 while split rows fell 21 -> 8).

**Local selection: context-witnessed release** (`local_selection/release.py`, `context_refit.py`, `objective.py`,
`rounds.py`, `stage.py`). The block posterior splits the carriers of near-identical rows by frequency, so a row the
context claims kept fewer than two supporters of its own and was withheld wholesale although its alleles are shared
with witnessed carriers; snaps made such rows common (std: wholly withheld rows 12 -> 53 in the replayed blocks, 99.2%
of their 10,600 withheld alleles correct against the nearest true row). The probability test is unchanged. In the
feedback rounds an allele that passes it but not the support test is now also released when it equals the call of the
context founder that claims the row and the context refit itself released that call (posterior 0.99 and the refit's
support threshold of two, by carrier count or expected carrier mass, with carriers assigned by linkage from the
flanking blocks; `ContextProposal.witnessed`, which excludes calls the refit kept from its input and completion
fills). The refit reads the same block's reads, so this is conditional reuse of evidence with a better carrier
assignment, not independent confirmation. On identical selected fits (alleles rescued / wrong against the nearest true
row): backcross 368 / 16, f3 720 / 1, std 8 contigs 18,641 / 58 (withheld 85,819 -> 67,178; released errors 3,164 ->
3,222), K30 8,144 / 115 (withheld 44,504 -> 36,360; released errors 2,353 -> 2,468); over the four simulations 0.68%
of rescued alleles are wrong, 1.09% among rows nearest a true row with at most two sampled copies, and on K30 alone
1.4%. Matched one-to-one to true rows instead, backcross and f3 rescues have no errors; std 289 of 16,574 matched
cells and K30 216 of 8,144 are wrong, and at sites private to the matched founder 14 of 239 (std) and 14 of 25 (K30).
On three TroMau contigs the withheld alleles fall 25,620 -> 9,241 and 77 of 79 wholly withheld rows are released. The
incumbent's release is no longer reused in a round with context witnesses, or when it used them. A second rule pooling
the carriers of mutual nearest rows passed the same error allowances but, matched one-to-one to true founders,
released the common row's allele at sites private to the matched founder (std: 60 of 67 such rescues wrong); it was
left out as a scientific exception to those allowances.

Targeted replays of saved selections (no end-to-end runs), each block replayed under its own evidence and context
where two 2.11 runs selected different panels (units: block-rounds):
- std, 8 contigs, under a 1e-6 likelihood jitter: the 4 divergent block-rounds now select identical panels (2.11:
  none shared even K).
- std, 8 contigs, calibration redraw (191 block-rounds): same K 187 (2.11: 177); identical panels 49 (2.11: 0); gaps
  above 1 nat to the better panel 0 (2.11: 49).
- TroMau, calibration redraw (1,217 block-rounds): same K 511 (2.11: 404); identical panels 164 (2.11: 0). Most
  remaining TroMau differences are likelihood-driven search differences under different candidate banks and contexts.

One claimer per assembled founder, against simulation truth, all other inputs fixed. Columns: block-rounds with equal
founder rows; changed selections; K toward / away from the true local founder count; summed Hamming distance from
true rows to the panel over the changed selections.

| Dataset | Block-rounds | Changed | K toward / away | Summed Hamming |
|---|---|---|---|---|
| f3 | 3,842 | 53 | 48 / 5 | 369 -> 0 |
| backcross | 2,967 | 13 | 10 / 3 | 47 -> 3 |
| std, 8 contigs | 6,404 | 131 | 118 / 12 | 1,787 -> 661 |
| K30 | 290 | 253 | 249 / 0 | 18,761 -> 13,110 |

The strongest extra rows are real shared recombinant haplotypes: one f3 example exactly matches 19 sampled
haplotypes that carry one true breakpoint (a descendant recombinant, not an original founder). On TroMau the change
adds local rows in about 2,100 block-rounds.

Context snaps against truth, every covered block-round replayed without and with them (changed selections; better /
worse summed Hamming recovery; exactly recovered true rows lost / gained): std 8 contigs 1,422; 856 / 354; 223 / 646;
backcross 50; 35 / 7; 5 / 29; f3 5; 2 / 2 (summed Hamming 29 -> 45); 2 / 1; K30 374; 214 / 120; 174 / 153. Snaps turn
over rare founders (one or two sampled copies): on std they lose 18 of the 95 rare true rows recovered without them
(18.9%, against 0.19% of commoner rows) and gain 76, net +58; on f3 two two-copy rows are lost (chr9, chr19) and one
one-copy row gained; on K30 18 two-copy rows are lost and 25 gained. The rule fixed before these receipts (drop snaps
if losses concentrate in rare founders while gains do not) is not met.

**Known limitations.** Founder refinement's objective prefers identity-wrong (chimeric) routings in multi-generation
crosses: started from a nearest-local-row proxy of the truth routing on saved f3 and std inputs, the production polish
raises the objective while moving away from truth (f3 chr9: a mismatch-plus-missingness proxy of 180 -> 62,337.5),
because a crossover inherited by many descendants is charged once per descendant under independent per-sample paths,
so routing a founder along the shared recombinant can score higher (an objective/target mismatch). Bounded search also
misses better-scoring routings, so runs whose inputs differ slightly land in different optima; per-contig differences
in founder identity between runs reflect both, and upstream changes, including those above, can seed a different
optimum. Better optimisation alone does not guarantee historical-founder recovery. A shared-descent-aware objective or
an explicit identity-ambiguity release is a research direction (the tested greedy uniform-prior recombinant
constructions did not make the truth proxy highest-scoring); no validated objective correction is included in this
release. Painting (outside this release's scope) can also move founder identity between runs: on std chr20 the
painting-stage identity disagreement rose by 34.1k while founder refinement changed by 0.8k. Discovery's path
dependence under exact ties (140 sites on std under a 1e-6 jitter) is unchanged.

**Validation.** The single end-to-end run of this release, to painting, compared with 2.11. Simulations against
truth (WOM: truth-to-panel wrong plus missing alleles under one-to-one matching; called-allele errors by each row's
nearest true row unless stated):
- std, 8 contigs: WOM 452,468 -> 465,652 (+2.9%; five scored 2.11 calibration redraws spanned 418.7k-452.7k),
  nearest-row called errors -0.1% (one-to-one +1.3%), missing alleles 24,259 -> 32,043, label switches 727 -> 718.
  At painting the identity disagreement rises on chr20 by 34.1k and falls on chr18 by 25.2k; chr20 changes by only
  +0.8k at founder refinement, so most of that swing arises in painting. At local selection WOM is -6.8% and at
  assembly level 1 -3.0% (called errors -11%).
- f3: WOM 149,864 -> 224,108 (+49.5%), almost entirely chr9 (46,403 -> 120,893, already at founder refinement), a
  founder-identity optimum (Known limitations); at local selection -1.6% and assembly level 1 -4.8%; missing 329 ->
  318; switches 19 -> 17.
- backcross: WOM 1,259 -> 1,346 (+6.9%); called errors 949 -> 681 (-28%); missing 310 -> 665; switches 6 -> 4.
- K30 (against the 2.11-equivalent run): WOM 67,647 -> 62,529 (-7.6%); called errors 10,356 -> 3,690 (-64%); missing
  9,203 -> 18,813; switches 30 -> 24.
- Missingness: on std final assembly, all uncalled alleles were kept-site local-selection withholdings that
  completion had not filled. The probability-derived alleles would match the assigned true founder at about 70% and
  74% in the two diagnostic strata (one-bit probability below / at least 0.85), against 72% and 78% in 2.11. These
  results support withholding these uncertain cells; they do not establish an unchanged release calibration.
- TroMau (against 2.10 on the G0 pseudo-reference, not truth): W100 wrong -3,890 (-1.2%) and W100 wrong + missing
  -3,406 (-0.7%), inside the locked screen (<= +22,168 / <= +22,093) and below all six 2.11 redraws (+4,028 to
  +22,168); W0 wrong -12.0%; W100 missing +0.3%; per-contig churn 39,080 (2.11 redraws up to 46,610); largest
  adverse contigs chr9 +4,455 and chr7 +3,712.

TroMau is scored against the dependent G0 v1.4 pseudo-reference, not truth, against 2.10, with 2.11's six calibration
redraws as the locked comparison (primary screen W100 wrong <= +22,168 and W100 wrong + missing <= +22,093;
secondaries W0, missingness, per-contig deltas and churn). An exceedance triggers attribution to a stage, not
automatic rejection, and a result inside the six-redraw envelope is not proof of accuracy or equivalence.

**2.11 validation exception.** After the 2.11.0 release its TroMau run exceeded a rule built on one 2.10 calibration
redraw. Six calibration redraws of 2.11 later showed changes of that size within redraw variability (W100 wrong
+4,028 to +22,168 against 2.10). 2.11 remains the operational baseline; the six-redraw cohort above replaces that rule.

## 2.11.0 (26 October 2026)

This release reworks the evidence stage's numerical fits, calibration sample and edge cases, and speeds discovery's
fixed-K fits. The site filter's artefact mixtures are fitted exactly and much faster on clean data; the read model is
calibrated on about one kept site in eight chosen by position instead of by index; contigs with no fitting data,
constant coverage or very short histograms no longer fail or produce NaN parameters; short contigs are calibrated; and
evidence-stage ties are reported. Results change: the calibration sample is new on every contig, and five site-filter
decisions that sit at a posterior of 1/2 move on TroMau.

**Read-model calibration sites** (`evidence/read_model/calibrate.py`, `evidence/config.py`): each kept site is used
for calibration with probability 1/`ReadModelConfig.marker_stride` (8), decided by a fixed pseudo-random hash of its
position alone, instead of every eighth kept site by index. Under the index stride one added or removed kept site (for
example one site-filter decision) changed the calibration sample for the rest of the contig; now it changes no other
site's use (the fitted model, and so the likelihoods, can still change across the contig). The sample is still a
uniform draw from the kept sites, of the same expected size (TroMau chr16: 9,711 of 78,415 kept sites; the stride took
9,802). On TroMau the per-sample error rates fitted on the new sample differ from 2.10's by no more than a redraw of
2.10's own sample does (mean change on the logit scale over 22 contigs -0.017 +- 0.022; 2.10 with another stride offset:
+0.002 +- 0.021), and the selected model changes on five contigs (chr4 and chr14 to the sample binomial; chr10, chr19
and chr22 to the sample joint model; the stride redraw changes seven). A thinning to sites at least 1 kb apart was tried
first and rejected: it samples along the genome rather than among the kept sites (28-33% of its sites had at most two
kept sites within 1 kb, against 11% of kept sites), and its fitted error rates were systematically lower (20 of 22
TroMau contigs; mean -0.14 +- 0.03).

**Site-filter mixture fits** (`evidence/site_quality.py`): the artefact M-steps of r3 (negative binomial) and r4
(beta-binomial) maximise the weighted artefact log likelihood over their two parameters jointly within the dispersion
clips, by projected Newton steps with analytic gradients and Hessians; each M-step stops when the full Newton step is
at most 1e-8 of max(1, |q|) on the log/logit scale and is certified stationary when its projected gradient per unit
artefact weight is at most 1e-9 (else recorded as a small-step stop); outcomes are recorded per mixture (`m_steps`).
Releases up to 2.10 searched the unbounded parameters with Nelder-Mead and then clipped the dispersion alone, so the
stored rate or mean could miss the maximum by more than the EM tolerance, at about 100 likelihood passes per EM
iteration. Where reads hold no artefact class, both EMs ran to their 300-iteration cap: on a clean simulated 329,000-site
contig with the filter on (N = 160) the filter took 866 s, now 14 s; on a 1.5-million-site contig over 25 minutes, now
86 s. On TroMau the r3 flag moves at one site on each of chr1, chr6, chr7, chr16 and chr19, each with a clean posterior of
about 1/2 (0.49995 to 0.50002); three become kept sites.

**Edge cases** (`evidence/site_quality.py`, `evidence/sample_quality.py`, `evidence/reports.py`):
- No reference site with depth heterogeneity (for example identical coverage everywhere): r2 is inactive with a reason
  (was an IndexError).
- No site eligible for r3 or r4: the rule is inactive with a reason, and the sample QC uses the configured
  classification heterozygote balance, labelled `h0_source` (was NaN parameters, a NaN QC balance and 300 empty
  iterations).
- Depth and heterogeneity histograms with fewer bins than the smoothing width, or with the mode at an end: smoothing
  covers the actual bins and the half-maximum walk stays inside the histogram (was an IndexError, or a walk that wrapped
  to the far end and lost the upper bound); a side without spread is handled explicitly. Histograms with an interior
  mode and enough bins are unchanged.

**Calibration's diagnostic fold** (`evidence/read_model/calibrate.py`): a contig with training and selection
observations but no admitted fold-3 observation (for example one spanning fewer than three 1-Mb windows) is calibrated,
with its diagnostic score unavailable; 2.10 used the fixed binomial model.

**Evidence ties** (`reports/ties.py`): the ties report gains the evidence stage: tied histogram modes of the depth and
heterogeneity cores (`ev.depth.mode`, `ev.heterogeneity.mode`), sites whose equal REF/ALT totals leave the QC's minor
allele to the convention (`ev.qc.minor_allele`, counted once per site wherever either allele could give near-private
cells), and read-model selections with more than one candidate inside the one-nat band (`ev.read_model.select.band`).
On TroMau the band decides on 14 of 22 contigs (the simplest candidate is applied); no other evidence tie occurs.

**Discovery fixed-K fits** (`discovery/fixed_k/fit.py`, `discovery/fixed_k/update.py`): the assignment Viterbi carries
each bin's maximum as it updates instead of rescanning the states, and per-sample allele terms are recomputed only at
sites whose fast sum is not certified. Exact: TroMau evidence and discovery are identical on all 22 contigs; discovery
takes 5% less time.

**Resume:** the evidence and discovery code identities and the read-model configuration change, so an output root of
2.10.0 is refused (VCF input: CheckpointMismatch at the evidence identity; simulations: at the simulation record); run
2.11.0 into a new output root.

**Investigated, not changed:** a replacement heterozygote-balance rule (two beta-binomials) removes nearly every site of
clean data in negative controls and was not adopted; the shared-sequence discovery prior remains research; local
selection, assembly and founder refinement have no exact speedup above about 2% of TroMau's lane time (lanes already
keep 69-71 of 76 CPUs busy); f3's founder errors are chimeric labels at junctions where a recombinant haplotype shared by
many relatives, or a weakly witnessed rare founder, decides the labelling (a model-level question for a later release).

### Validation
- Site filter: on TroMau the M-step change moves five r3 decisions, all at a clean posterior within 5e-5 of 1/2; every
  M-step is certified stationary or stops on the step tolerance (none fails its line search or reaches the cap).
- Discovery fixed-K speedup: TroMau evidence and discovery identical on all 22 contigs.
- Calibration sample: sampling checks (membership depends on the position alone; rate 1/8 on random and regular
  position axes; on TroMau the sample has the kept sites' density composition) and the TroMau fit comparison above.
- Edge cases: fixtures for constant coverage, empty r2/r3/r4 domains, short and edge-mode histograms and contigs
  without diagnostic-fold observations.
- Resume: a 2.10.0 root is refused (VCF input at the evidence identity; simulations at the simulation record).
- End to end with the rejected 1-kb thinning (all other 2.11 changes as released), against simulation truth to
  painting: std 8 contigs, founder allele errors +7.0%, one chimeric relabelling on chr10 (2.10 with another stride
  offset: +7.75%, the same event); f3 identical; backcross 1,262 -> 1,260. End-to-end runs of the released calibration
  sample (simulations against truth; TroMau consistency with a pseudo-reference) were still running at release.

## 2.10.0 (26 October 2026)

This release keeps a run's CPUs busy without changing what it computes. CPUs are rebalanced dynamically at every level:
every piece of work runs its fair share, work above its share returns the excess at its next boundary while others wait,
and idle CPUs go to the work that can use them. Contig lanes' pool workers start with the package imported, compiled
kernels load from memory-backed storage, lanes are narrower, fewer workers are replaced, independent founder-refinement
scorings run concurrently, shadow families are scored in one pass, and local selection's copying model no longer pads
small sample sets to 64 lanes.

**Dynamic CPU rebalancing** (`execution/budget.py`, `execution/pools.py`, `execution/chromosome_workers.py`): one rule
at every level. A pool task or concurrent in-process task runs its fair share of its CPUs, floor(total / (running +
waiting)) plus the remainder in turn, capped where it cannot use more; a task above its share (it grew while it had
fewer peers) returns the excess at its own next boundary, and as tasks finish the others' shares grow. With 80 CPUs and
10 tasks every task runs 8, however they arrived (2.3 to 2.9 let a task that grew alone keep its CPUs while later tasks
waited for one). A pool whose total grows while tasks wait starts more workers (closed again after the batch); its total
follows its lane's lease both ways, re-read every `LIVE_TOTAL_SECONDS = 0.25` s. Contig lanes start with claims in
proportion to their contigs' work and then share the CPUs equally among the running lanes; a lane borrows idle CPUs (of
finished lanes, or of a lane waiting for memory) while no lane is short and returns them at its next boundary once one
is. Two variants were measured and rejected: shares that followed estimated remaining work starved the critical contig's
final refinement, and lanes that lent their CPUs over the serial steps between their parallel constructs got them back
only at the borrowers' boundaries (both 3-7% slower contig lanes on three-contig runs). A lane's coordinator holds no
CPU while it waits for its pool and takes one of the pool's CPUs while it works on a result. Discovery's evidence
helper, which prepares the next contig's evidence beside the block pool, holds a quarter of the pool's CPUs while it
runs instead of a second full budget. Checkpoint compression and decompression, painting's sample batches and
refinement's beam (one thread during its serial preparation) follow the live share. Claims are taken only from free CPUs
and returned by their holder at its own boundaries, so a running kernel never changes its team and the numerical teams'
claims stay within the run's ceiling (worker start-up and a lane's serial steps between parallel constructs hold the
claims they had). On a 64-CPU node, against this release's other changes alone (without rebalancing, the refinement
concurrency and the copying-model change below), contig lanes take 650 instead of 705 s on 8 simulated contigs (busy
CPUs in the lanes 52.3 -> 55.5 of 64) and 701 instead of 702 s and 485 instead of 500 s on two three-contig simulations.

**Local selection's copying model** (`local_selection/path_model.py`): the E-step lays samples side by side in lane
blocks; a block is now only as wide as its share of the samples needs, rounded up to `LANE_ALIGN = 8` (the number of
blocks is unchanged): 40 samples use 40 lanes instead of 64, 80 samples two blocks of 40 instead of 64 + 16. Lanes are
independent and totals add samples in order, so results are bitwise identical. One E-step at 40 samples takes 13-16%
less time (5-8% at 80); on 8 simulated contigs (64 CPUs) contig lanes take 634 instead of 698 s.

**Founder refinement concurrency** (`assembly/founder_refinement/count.py`, `stage.py`): the count pass's independent
scorings (death-class totals, clean-up and deletion-ranking scores, terminal-family scores), the death-class members'
decisions and the polish fits of equal-F alternatives run as concurrent in-process tasks instead of one after another;
the scorings and decisions run in waves whose estimated transient memory is at most `COUNT_FANOUT_BYTES` (4 GiB; a
single larger one runs alone), and results are collected in their original order. With rebalancing, five recorded
refinement components (N = 80, N = 320, K = 30) replayed together on a 64-CPU node take 347 instead of 409 s (-15.2%)
with identical results.

**Worker start-up in contig lanes** (`execution/bootstrap.py`): every child process applies the run's forkserver
preload list again. A contig lane is itself a forkserver child, and its own forkserver used to preload nothing, so
every pool worker in a lane started by importing NumPy, Numba, SciPy and the stage modules itself (about 8 s from
a warm filesystem, 34 s cold). On a 40-sample, 4-contig simulation with 64 cores, contig lanes take 385 instead of
436 s (-11.8%), local selection's lane time falls by 26%, and busy cores in the lanes rise from 43.7 to 46.9 of 64.
Compiled kernels still cannot be preloaded: loading a parallel kernel starts the OpenMP thread pool, after which a
forkserver must not fork.

**Kernel cache placement** (`execution/bootstrap.py`): every new pool worker loads its compiled kernels from the
run-local cache. By default that cache is now in memory-backed `/dev/shm` when it is writable with at least
`KERNEL_SCRATCH_FREE_BYTES` (4 GiB) free, unless a Slurm job provides `SLURM_TMPDIR`; otherwise the system temporary
directory, as before. The memory-backed directory is private to the run, named after the process that created it and
removed when that process exits; a later run on the host removes one whose creator has ended, before it checks the free
space (a job's scratch or the temporary directory is kept, as before). An explicit `NUMBA_CACHE_DIR` is honoured. With
2.9.0's code, moving the cache from a shared network filesystem to `/dev/shm` made the same simulation 12.5% faster (500
instead of 572 s).

**Narrower contig lanes** (`execution/contig_lanes.py`): `auto` lanes get one lane per `CONTIG_LANE_CORES = 8`
cores (2.9.0: 16). On 8 simulated contigs with 64 cores, lanes take 736 instead of 803 s (-8.4%) with less total
CPU; more contigs run at once, so peak memory rises (see Validation).

**Fewer replaced workers** (`assembly/hierarchy/level.py`): hierarchy pools no longer retire workers after four
batches. Workers are replaced when their private memory grows past the pool's threshold, as in the other pools;
where private memory cannot be measured (no `/proc/self/smaps_rollup`), the four-batch cap remains.

**Serial sections in local selection** (`local_selection/context_refit.py`, `local_selection/stage.py`,
`pipeline/contig_work.py`): context refit finishes each component's proposals in its pool task instead of in the
coordinator, final assembly uses the panels local selection has just returned instead of reading them back, and
the proposals between rounds are written with the lane's thread share. Lanes take 1.3% less time.

**Founder refinement** (`assembly/founder_refinement/`):
- Shadow families are scored with one forward pass per family (`count.py`), bounded by a 1 GiB family-table
  budget with the per-sample path as the fallback. On five recorded refinement components, refinement takes
  265 instead of 310 s (-14.5%; an 80-sample level-3 component 23 instead of 58 s).
- Long repairs and bridges re-read their thread share at each chunk, and window templates run largest first
  (`search.py`).

**Portability:** comments and documentation no longer name a particular cluster, filesystem or service level.

**Resume** (`pipeline/driver.py`, `pipeline/contig_work.py`): records a run reuses without running their stage
(completed discovery, local selection, final assembly and painting) are now checked against the current code and
configuration first, as the stages check them; 2.9.0 skipped that check for completed stages, so a completed output root
of another version could be reused unchecked. The discovery, local selection and assembly code identities change, so
an output root of 2.9.0 raises CheckpointMismatch at discovery; run 2.10.0 into a new output root.

**Failing fast** (`execution/pools.py`): a pool worker that dies during a task (killed, out of memory, crashed) now
fails its batch within about a second, naming the worker and its exit code; 2.9.0 replaced the worker silently and its
consumer waited for the lost result indefinitely. Closing or terminating a pool also ends a wait for its results.

**Measured, not changed:** grouped candidate-move scoring, shared boundary messages and guard shortcuts in
founder refinement (exact, but 0.4% on five recorded components: within noise); keeping each lane's pools alive
between steps (2.3% less lane time, but 24.7 GiB more peak memory on 8 contigs and two defects found in review);
short OpenMP spin-waiting between parallel regions and capping refinement's acceptance-pass team (no gain);
splitting evidence preparation from discovery's workers; narrower sample lanes in path fits; dispatch look-ahead;
precomputed item dosages in refinement (less than 0.3% of its CPU). Each was exact and gained nothing measurable,
or was slower.

### Validation
Every test compares the released records (called and released founders, assembly completion and final panels,
paintings, local selection and discovery blocks) value by value. Timed comparisons run both arms back to back on the
same node; the capped 320-sample comparison uses the contigs both runs completed.

| Test (2.10.0 against 2.9.0; exclusive 76-CPU node unless noted) | Outputs | Contig lanes, 2.9.0 -> 2.10 |
|---|---|---|
| 40-sample simulation, 8 contigs, 2.10 run first | identical (98.1 M values) | 770 -> 595 s (-22.7%), 72% -> 84% busy; whole run 970 -> 804 s, CPU -8.2% |
| the same, 2.9.0 run first | identical | 775 -> 597 s (-23.0%), 72% -> 83% busy; whole run 955 -> 784 s, CPU -9.0% |
| K = 30 simulation, 5 contigs | identical (21.1 M values) | 1,357 -> 1,112 s (-18.0%), 69% -> 81% busy; whole run 1,516 -> 1,255 s; peak live memory 55.0 -> 57.6 GiB |
| three-generation families, 3 contigs (64-CPU node) | identical (64.3 M values) | 778 -> 691 s (-11.1%) |
| backcross, 3 contigs (64-CPU node) | identical (55.3 M values) | 624 -> 487 s (-22.0%) |
| refinement replays: 5 components at N = 80, N = 320 and K = 30 (64-CPU node) | identical (panels, scores, truth) | batch 474 -> 382 s (-19.5%), CPU -8% |
| TroMau cross (real data, preset `tromau`), 22 contigs to painting | identical (189.9 M values) | 5,162 -> 4,478 s (-13.3%), 83% -> 92% busy; whole run 5,779 -> 4,982 s, CPU -3.1%; peak live memory 41.6 -> 54.4 GiB |
| 320 samples, 22 contigs, each run stopped after 4 h | identical: discovery of all 22 contigs, released records of the 8 both completed (449 M values) | not compared (capped runs) |

The release code itself (with the resume and fail-fast fixes) repeated the first test on the other dedicated node:
identical, contig lanes 784 -> 603 s (-23.1%), 71% -> 83% busy.

Before rebalancing, the same changes without it (concurrency fixes only) gave 805 -> 638 s and 790 -> 633 s on the
8-contig test (both orders, -20%) and 1,335 -> 1,195 s on K = 30, peak memory 56.9 -> 57.8 GiB.

The same 8-contig test with all kernel caches in memory also ran 2.7.1's admission (the 2.8 development baseline
"f35"): its lanes took 785 s, so 2.8.0's slower full-node runs relative to it no longer appear (2.9.0: 766 s).

## 2.9.0 (25 October 2026)

This release changes how three parts of a run are scheduled, and nothing they compute: memory admission scans its
task table only when it needs to, contig lanes take their contigs from one shared queue, and discovery runs every
contig's blocks from one queue without draining its pool between contigs.

**Memory admission** (`execution/memory.py`): the scan of the 4,096-slot task table for running and queued work runs
only on the progress-exception branch (a head waiter that does not otherwise fit), not on every admission. Decisions
are unchanged. Contended admission takes 54 instead of 95 microseconds per call (-44%), with 36% less CPU.

**Contig lanes** (`execution/chromosome_workers.py`, `execution/contig_lanes.py`): each lane starts with one of the
longest contigs, and a lane that finishes a contig takes the next from one queue, longest first; 2.8.0 fixed each
lane's contigs at the start. Lease growth counts a queued contig's own work, and a recycled lane worker's
replacement continues from the queue. The lane log line names the starting contigs and the queue, and a failure
message lists each lane's contigs and the contigs no lane started. Replayed on recorded lane times, TroMau's lanes
take 5,225 s instead of 6,100 s (-14%) and the 22-contig N = 320 simulations' 4-10% less. With few contigs whose
work estimates are mis-ranked it can be slower (+3.6% on a 4-contig test).

**Discovery** (`discovery/stage.py`): one pool batch takes every contig's blocks in contig order, so the next contig's
blocks start as workers free up from the current contig's last ones; 2.8.0 drained the pool at the end of each
contig. Evidence is still prepared one contig ahead on one helper thread (`discovery.stage.PREPARE_AHEAD = 1`). On a
failure, the completed blocks of every unfinished contig are kept, so a rerun discovers only the missing blocks, as
before. On 8 TroMau contigs, discovery takes 5% less time.

**Resume:** the discovery package's code identity changes, so an output root of 2.8.0 raises CheckpointMismatch at
discovery; run 2.9.0 into a new output root.

**Measured, not changed:** checkpoint writes (about 0.2% of lane time); evidence site-quality fits (no exact
arithmetic speedup); memory admission at N = 320 (2.8.0 paces nothing there: 107,884 admissions, none paced); and
the fixed four-batch worker cap of assembly pools (no measurable gain). 2.8.0's measured admission removes 2.7.1's
false pacing on partial nodes (a 64-core node with about 211 GiB: 2.7.1's governor paced 15,015 tasks on 260 GiB of
predicted peaks with 23 GiB in use; lane phase 1,261 s with it, 770 s with 2.8.0). Why 2.8.0's whole runs are slower
on full nodes is left to the 2.9 whole-genome matrix.

**2.8.0 notes corrected:** their whole-run times are runs to the end of painting, not founder refit.

### Validation
Each change was tested alone against 2.8.0, both arms back to back on one node, with the outputs compared record by
record: called and released founders, assembly completion and final panels, paintings, and local selection and
discovery blocks by index.

| Change | Test | Result |
|---|---|---|
| Memory admission | 80-sample simulation, chr19 | identical (15.7 M values, 2,728 blocks) |
| Contig lanes | 40-sample simulation, 4 contigs, 2 lanes | identical (46.1 M values, 11,248 blocks) |
| Discovery | TroMau, 8 contigs, to discovery | identical (3,999 blocks, 5.5 M values) |
| All three | 40-sample simulation, 4 contigs, 2 lanes, 32 cores | identical (46,087,180 values, 11,248 blocks; largest drift 0) |

Together, timed on the same test with 64 cores, both arms back to back on one node, a run takes 789 s instead of
818 s (-3.5%): contig lanes 617 instead of 642 s, discovery 90 instead of 94 s. Peak memory is 17.4 GiB (2.8.0:
17.2 GiB), no admission is paced, and the results are identical. Four contigs on two lanes leave the lane queue
little to reorder; the replays above estimate its effect on whole genomes.

- A discovery failure part-way through a contig keeps the same completed blocks as 2.8.0, and the rerun discovers
  only the missing ones.
- Not run: the whole-genome matrix (TroMau, the 22-contig simulations, std, stress, f3, K30) on full nodes.

## 2.8.0 (25 October 2026)

This release rebuilds founder-path refinement in assembly: a robust maximum-path objective of a per-homologue
founder-switching model with map-derived switch costs that reads partially called sites, a symmetric count null,
polynomial per-sweep bounds on every search step, and a final polish of the decided panel with a wider move set.
Outside refinement it adds measured-memory admission, full linker EM steps, a stall watchdog and bounded discovery
and join proposals (**Outside refinement**, below).

**Why.** 2.6/2.7.1 refinement scored panels with a uniform 20-nat penalty per diplotype change and read only sites where
every local row was called; at K = 30 that ignored 18-25% of a component's sites, and the count step missed a founder
the data support (a K = 30 simulation component with 30 true founders: 28 called, 19,744 wrong-or-missing alleles;
2.8 calls 29, 12,589). Its window
programmes had no polynomial bound, and its count null depended on which panel was the incumbent.

**Objective** (`objective.py`, `workspace.py`):
- Each homologue follows its own founder chain; between a sample's consecutive informative sites it resets with
  probability rho = 1 - exp(-h), h = `refinement.reset_rate` (1 per Morgan per homologue) times the genetic-map length
  crossed, and a change costs w = log(1 - rho (1 - 1/K)) - log(rho / K) per homologue. Exact on the K(K+1)/2 unordered
  pairs, O(Q + K) per site. Panel-independent normalisers are added where fits of different K are compared.
- Sites: kept discovery sites with at least one called local row (2.7.1: every local row called). A pair with an
  uncalled row emits the predictive class of the unknown allele(s) under a uniform allele prior; the same uncalled
  row twice is one latent allele.

**Search** (`acceptance.py`, `search.py`, `proposals.py`):
- At most four full re-rankings and four suffix exchanges per junction and pass, then one-at-a-time re-scoring in the
  last ranking's order: every pass is at most cubic in K (2.7.1's rescoring was quartic in the worst case). The
  restarted window programme keeps at most 512 children per unit. Binding bounds are counted and warned.
- Lexicographic (guard) messages: a search's first sweep, and every sweep after one that read switch costs, is
  lexicographic; other sweeps run plain and are rerun lexicographically at the first coarse move above the
  tolerance. Decisions are identical to fully lexicographic sweeps.

**Count step** (`count.py`): the sequence part of every decision uses the larger panel's own canonical death class
(every member evaluated; disagreement or an unresolved member gives the smaller K with ambiguity), so the same two
endpoints get the same call whichever was the incumbent. Accepted additions need no check deletion round; no count
cycle arises.

**Polish** (`stage.py`, `search.py`, `beam.py`): after the count pass, the decided panel and its same-K equal-F
alternatives are refitted once more (one forward-first search) with three move families at every fixed point: a window-beam escape on a
map-aware surrogate, long one-founder moves (64-block beam offers, width 16) with one private correction, and
fine-window/suffix bridges. The polish
never changes K; count decisions are not re-evaluated on the polished panel (recorded). A polish that reaches the
window work bound keeps the count pass's panel.

**Records:** refinement records gain `polish` (status, pre-polish fit and F, gain); fit labels `polish`,
`polish.alt<i>`; new work counters (`content_rounds`, `round_cap`, `stale_evaluated`, `exchange_cap`, `lex_items`,
`lex_abort`, escape and long-repair counters). `count_cycle` is always false.

**Resume:** the discovery, local selection and assembly code identities change, so an output root of an earlier
version raises CheckpointMismatch at discovery; run 2.8.0 into a new output root.

**Outside refinement** (integrated from the 23 October 2.8 components and validated together with the refinement;
see Validation):
- **Linker:** full regularised EM steps (`AssemblyConfig.linker.learning_rate_decay = 1.0`). Uniform initialization,
  the 50-update cap, the fixed-point residual tolerance and the regularisation are unchanged. Finite-iteration
  transition estimates and downstream decisions may change.
- **Measured memory admission:** work is admitted from actual job memory pressure, with no reservations or predicted
  peaks. Below 80% of capacity outer starts are unrestricted; from 80% to 90% each waits for a full observed 0.5 s
  sampling interval; above 90% only the progress exception admits new outer work. Nested continuations proceed so
  admitted work can finish. Scheduling rules only; a delayed or indivisible allocation can still exceed capacity.
- **Tidy-up and watchdog:** lossless int8 sparse flip indices and sizes in completion and founder-allele posteriors
  (K <= 30); unused dense diagnostic helpers removed; abandoned optional memo requests released promptly. A watchdog
  fails a run after 30 minutes of frozen grant/start/release counters with waiters and under one CPU core of activity,
  keeping checkpoints.
- **Discovery proposals:** maximal-clique enumeration is replaced by a content-ordered packed edge-clique cover. Every
  compatible pair is covered before the existing size-class budget, applied per component; larger compatible groups
  can be omitted. The unchanged downstream scoring decides which proposed rows survive.
- **Join proposals:** at most q paths per endpoint, cut ties split in canonical content order; constant-transition
  families use a covering set instead of a Cartesian expansion; the beam-materialization/unjoined fallback is removed.
  These are declared proposal approximations and can change selected or released founders.

### Validation
Component replays: 36 native components, including the six pieces of the equal-resource benchmark, refined from
the same incoming panels as 2.6:

| Component | 2.6 | 2.8 |
|---|---|---|
| K30 m5 (L2; truth 30 founders) | 19,744 / K28 | 12,589 / K29 |
| f3 (L4) | 136,843 | 135,891 |
| stress chr19 [460, 904) | 3,493 | 731 |
| intermediate_K | K7 | K6 (truth 6) |
| K30 L1 / L2 | 2,846 / 11,428 | 2,843 / 11,425 |
| K30 L3 (also the k30_m2 diagnosis ID), m4 | 5,048, 19,187 | 5,053, 19,191 |
| all other 27 component IDs | | same WOM and K |

Validation does not meet the frozen release requirements: f3 per-case and pooled whole-pipeline WOM, fixture trap
rescues and fresh identifiable harms, and per-component count-unverified outcomes (rule 4) fail.

Refinement time on the equal-resource benchmark: 1.93-1.96x 2.6 in two measurements (limit 2x). On whole-pipeline runs
with production warm-up and 2.7.1's memory admission, refinement level walls summed over contigs are 2.14x 2.7.1's
(std 2.47x, stress 2.11x, f3 2.04x, K30 1.92x; backcross 2.10x and TroMau 2.26x measured on a near-identical variant).
Whole runs (to the end of painting) take 1.17-1.58x 2.7.1's with this release (std 1,953 s vs 1,236 s, stress
1,751 vs 1,131, f3 1,823 vs 1,173, K30 1,765 vs 1,512) and 1.14-1.34x with the refinement alone: measured-memory
admission does not hold these runs back (no paced admission without memory pressure), but cores stay idle for
longer; the cause is not yet isolated. The 22-contig simulations (refinement alone) take 1.30x and 1.70x 2.7.1's.

Founder-only whole-pipeline matrix (runs stop after founder refit), founder_refit wrong / missing alleles:

| Case | 2.8.0 | 2.6 |
|---|---|---|
| std | 7,120 / 5,208 | 8,385 / 5,207 |
| stress | 1,372 / 2,154 | 3,914 / 2,152 |
| backcross (refinement alone) | 12,409 / 3,927 | 21,518 / 3,926 |
| K30 | 9,068 / 57,203 | 9,109 / 57,142 |
| f3 | 303,129 / 778 | 282,150 / 775 |
| 22 contigs, seed 10000 (refinement alone) | 162 / 159 | 953 / 159 |
| 22 contigs, seed 10001 (refinement alone) | 117 / 17 | 117 / 17 |

- The 22-contig simulations have 320 samples (20 F1, 100 F2, 200 F3). Seed 10000 differs from 2.6 and 2.7.1 on one
  contig (chr3: 1,044 to 253 wrong or missing); K = 6 on all 22 contigs of both seeds, as in 2.6 and 2.7.1.
- The outside-refinement changes alter founder_refit wrong-or-missing alleles by +1 (std), 0 (stress), -179 (K30)
  and +5 (f3); final K is unchanged on every contig.
- f3 is 7.4% worse: on chr2 two founder rows are relabelled across one junction, a labelling that a crossover
  shared by 21 related samples makes the objective prefer (the correct labelling scores 184 nats lower). 2.7.1's
  search reached the correct panel through a different join.
- No new false parent edges (std, stress, K30 and f3 with this release; backcross with the refinement alone). TroMau
  (refinement alone) keeps 2.7.1's founder count on all 22 chromosomes.
- Forced memory pressure (std with a 40 GiB limit, against an unforced peak of about 68 GiB): identical results and no
  stall; the peak reached 44.3 GiB, because admitted work is allowed to finish.
- Fixtures: trap rescues 20/36; identifiable correct-start harms 0/123 (original) and 1/118 (fresh); all
  correct-start harms 99/528 and 81/512 (2.7.1: 233/528 and 220/512); F72 72/72.
- Not rerun with the outside-refinement changes: backcross, TroMau and the 22-contig simulations. Not run: held-out
  simulations, TroMau pseudo-reference consistency.

## 2.7.1 (22 October 2026)

This release restores founder-path refinement in assembly, which 2.7.0 removed.

**Why.** With refinement off, the 2.7.0 founder-only validation on simulations with known truth failed its gate. Pooled
native-refit founder wrong-or-missing rose from 394,278 to 671,099 (+70%; gate at most +1%):

| Case | 2.6 | 2.7.0 | Change |
|---|---|---|---|
| std | 13,592 | 30,670 | +126% |
| stress | 6,066 | 81,868 | +1,250% |
| f3 | 282,925 | 437,777 | +55% |
| backcross | 25,444 | 53,668 | +111% |
| K30 | 66,251 | 67,116 | +1.3% |

- **Cause:** surplus founder rows left by the joins survived. Several chromosomes ended with K = 7-9 where the truth
  has 6, which split true founders across rows; refinement's count step removes them.
- **f3:** gained one new false parent edge.
- **TroMau:** native pseudo-reference mismatches fell (615,274 -> 470,063). That is consistency with a dependent
  pseudo-reference, not accuracy: extra founder rows match its eight G0 haplotypes more easily.

**What 2.7.1 keeps from 2.7.0:**
- Founder completion's exact O(K^3) reorganisation, with identical released results. W3's identity runs on std, K30
  and TroMau were identical with refinement on.
- The linker EM update cap of 50. Fits that stopped at the old cap of 20 can change.
- Deterministic seed-family promotion order in joins, and removal of the unused `shadow_bounds._join` and
  `join_envelope`.

**What 2.7.1 restores from 2.6:**
- The 2.6 refinement model, with its count step, records and reports: `founder_counts.tsv` comes from refinement's
  count pass again.
- The 2.6 documentation of these.

A new refinement is being designed for 2.8.

**Resume:** assembly code identity changes, so 2.7.0 output roots are recomputed from assembly onward, not resumed.

**Operational fix: a memory-governor deadlock in founder refinement.**
- **Symptom:** 22-contig runs with refinement on could stall indefinitely. Reservations sat near the limit while real
  memory was free, and waiters asked for 0.01-0.03 GiB with almost no CPU in use.
- **Cause, present since 2.6:** a concurrent task whose memory was granted while it still waited for a CPU slot was
  marked as running work. That disabled the governor's rule of admitting the head of the queue when every holder is
  waiting.
- **Why 2.7 exposed it:** larger refinement memory estimates (the guard traceback term) and long-held worker
  reservations made the state reachable.
- **Fix, scheduling only:**
  - granted-but-unstarted tasks stay queued until they start (`execution/budget.py`, `execution/memory.py`);
  - guard scratch is reserved only around guarded acceptance (`assembly/founder_refinement/{search,acceptance}.py`).
  Both change when work is admitted, never what is computed. The guard-scratch change was checked identical on
  backcross, N640 and TroMau, and a bounded reproducer hangs before the fix and completes after it.

**Known limitation, unchanged:** family phase, the recombination maps derived from it and phased-sample VCFs are known
to be broken and are not validated.

### Validation

Founder-only matrix (runs stop after founder refit; family phase excluded), against 2.6. Each cell is founder_refit
wrong / missing alleles.

| Case | 2.7.1 | 2.6 |
|---|---|---|
| std | 8,385 / 5,207 | identical |
| stress | 3,914 / 2,152 | identical |
| backcross | 21,518 / 3,926 | identical |
| s10000 (22 contigs) | 953 / 159 | identical |
| s10001 (22 contigs) | 117 / 17 | identical |
| f3 | 282,249 / 770 | 282,150 / 775 |
| K30 | 9,048 / 57,339 | 9,109 / 57,142 |

- Every s10000 and s10001 chromosome has K = 6 in one piece.
- Pooled wrong-or-missing over the seven simulations: +0.06%. No new false parent edges; f3 keeps 2.6's eight.
- The f3 and K30 differences come from the linker cap of 50.
- TroMau native pseudo-reference mismatches: 596,014 against 615,274 (−3.1%; consistency only, not accuracy).
- The governor-deadlock fix changes scheduling only. std and TroMau rerun with it are identical, per chromosome, to the
  matrix runs above.

## 2.7.0 (22 October 2026; superseded by 2.7.1, do not use)

This release retains founder completion, removes founder-path refinement from assembly, and raises the linker EM
update cap to 50. These owner-directed changes can alter scientific results. Final founder validation is pending.

**Known limitation:** family phase, the recombination maps derived from it and phased-sample VCFs are known to be
broken and are not validated in 2.7. Those stages still run by default, unchanged from 2.6. Validation stops after
founder refit and assesses per-chromosome founder concordance, parent edges and TroMau pseudo-reference consistency.

### Assembly and count reporting

- Joined components and their own release alternatives pass directly between assembly levels and into the final
  assembly product. Founder-path refinement is disabled; its source is retained outside the pipeline and kernel
  warm-up for the 2.8 rebuild. Downstream founder allele refit remains active and is a distinct stage.
- Count reports use the latest actual join, distinguish incomplete verification from demonstrated ambiguity, and
  identify local panels that have never joined. The latest join is included only once in cycle totals. There is no
  new final count-addition pass or heuristic dead band for near-zero count margins.
- The final founder evaluation stage is `assembly`. Assembly schema 3 and its stored configuration reject refined
  checkpoints before completed-stage shortcuts, including the superseded schema-3 completion-drop candidate.
- Ambiguity records from every accepted seed reaching the selected canonical panel are retained. Accepted-seed
  histories are promoted in founder-path order, so thread completion order does not order their witness families.
  Two unused shadow-count helpers were removed; the separate full-site proof remains.
- Discovery I6 remains a documented limitation, not a new inference rule. No unobserved founder allele is selected
  by a new convention in this release.

### Exact computational changes and limits

- Packed symmetric emissions, reused splice and count-cover calculations, compact beam families and retained-work
  memory accounting reduce selected costs without changing their numerical objectives. Refinement-only speed and
  window-policy work is inactive; no refinement window limit governs the shipped assembly path.
- Completion represents growing support with sparse flips instead of a configuration-by-founder matrix. At fixed
  sample count and block length it is O(K^3); exact cubes with at most ten uncertain alleles are bounded by 1,024
  configurations. The small-cube cavity cancellation fallback is O(N 2^U K^2). Independent focused benchmarks fit
  EM dosage exponents 4.28 to 2.89, configuration updates 4.03 to 2.68 and cross-fit blocks 3.33 to 1.99. No measured
  memory reduction at K<=30, or universal whole-pipeline speedup, is claimed. Full completion identity gates remain
  pending until the std, K30 and TroMau comparisons finish.
- Complete two-row exclusion domains use adaptive segment evaluation or direct forwards. At production bin counts
  and K<=30 the direct route is used, and this term grows approximately as K^4 until K^2 is roughly 3m or larger.
  The cubic segment route does not establish a cubic bound for every production workload or the whole join search.
- Any binding beam allowance can change outputs. A constant family beyond the allowance releases the unsearched
  anchor panel without masking, flagged in metadata; a nonconstant family retains the incumbent or leaves the inputs
  unjoined. Earlier validation observed no binding. This remains a quality risk for the 2.8 work; it is not an exact
  alternative to searching the domain.
- A packaged kernel manifest supplements learned store signatures, compiling release overloads once in the warm-up
  pool. Disabled or removed kernels are skipped and the 2.6 cache-signature check remains. The 508.9 s cold N80 result
  belongs to c0 plus the W2 manifest change, not this final tree; no final-tree cold timing is claimed.

### Linker convergence

- The residual is checked at the returned transition matrix, adding an E-step per fit. W4's bounded fixtures used
  2% to 29% more updates under that stopping rule. This correction can change outputs.
- The owner raised the update cap from 20 to 50. Tolerance 0.001, initialization and damping remain unchanged; at most
  51 E/M evaluations include the returned-iterate check. A fit at the cap may still be unconverged, and its residual
  is reported honestly. On one saved K30 gap, the longer fit converged at update 34; this is not a whole-run guarantee.
- Cap hits were common in the earlier cap-20 c2 keep matrix: 11,187/24,170 TroMau gap fits (46%), 417/561 K30 fits
  (74%) and 6,470/24,875 std fits (26%) stopped above tolerance. These are not cap-50 frequencies. The final matrix
  supplies the cap-50 census.

### Validation and deferred work

- Final founder-only runs on this refinement-off tree FAILED the pooled gate (+70% founder wrong-or-missing; see
  2.7.1). Gates are pooled native-refit founder wrong-or-missing
  at most +1%; no simulation's native wrong calls increasing both >5% and >200 cells; no new false parent edges;
  and TroMau native-refit pseudo-reference mismatches at most +1%. TroMau consistency is not accuracy against known
  parents or true founder haplotypes. Offspring genotype and phase-switch gates are excluded by the owner.
- Per-chromosome founder concordance with the true generating haplotypes, s10000 and s10001 (22 chromosomes each, 6
  true founders): with refinement on (the earlier candidate), every chromosome is one piece with all 6 founders found
  and matched, and 953/159 (s10000) and 117/17 (s10001) wrong/missing alleles of 53,741,112, identical to 2.6.
  The refinement-off tree did not reach these: see 2.7.1.
- The earlier f3_fam matrix inherited eight false parent edges unchanged from 2.6. They were not new errors; the
  final matrix must still report their identities and any new edges explicitly.
- Export manifests contain wall-clock release diagnostics and are not byte reproducible; those diagnostics do not
  alter VCF content. This does not validate the known-broken phased-sample outputs.
- Release continuity, shared-junction effects in joins, missing K30 founders without additions, refinement redesign
  and linker acceleration are deferred to 2.8. No W4 continuity fix or refinement-memory overlay is included here.

## 2.6.0 (21 October 2026)

Release 2.6 speeds discovery and local selection, and separates the copying intensity from assembly's gap horizon.
Both parameter defaults stay at 3, preserving scientific results; the local-selection conventions remain unchanged.

### Local selection: conventions checked

- The bank's 512 + 64K start cap binds in six block-rounds on TroMau and K30. There are 73 same-K
  distinct-winner ties, six of which advance the inner search.
- Reversal checks change no released local call, K, missingness, support or decision in 223 flagged
  block-round replays or through all three rounds on K30 m1 and TroMau chr7. One uncalled latent cell
  and posterior drift up to 7e-6 remain on K30. No remedy is needed on this evidence.

### Copying and assembly parameters separated

- The shared `n_generations` is replaced by two independent internal parameters:
  `LocalSelectionConfig.model.redraw_rate_per_morgan = 3.0` is the founder-copy redraw intensity per Morgan;
  `AssemblyConfig.gap_horizon_scale = 3` limits linker and beam proposal gaps. Defaults retain the same
  arithmetic and scientific results. Neither parameter is a public TOML setting.
- In a bounded chr19 sweep, copying intensities 1 and 6 worsen TroMau G0 pseudo-truth consistency.
  Other settings bring no principled gain across std, F3 and TroMau. Both defaults stay at 3;
  no generation labels or painting-switch estimates set them.

### Discovery speed

- Birth proposals retain canonical keys and descriptors until fitting needs panel arrays. Fractional-founder
  scalar certification runs before term-column views are built. Search budgets, objectives and fits are unchanged.
- Matched single-contig discovery wall times on 76-core Ice Lake nodes fall from 33.3 to 30.1 s at N=80
  (-9.6%), 54.4 to 47.0 s at N=320 (-13.6%), and 32.7 to 30.6 s on TroMau (-6.4%). All 2,696 blocks
  and fit counts agree, with zero measured continuous drift. Each result is one matched pair, not a
  whole-pipeline timing estimate.

### Local-selection speed

- Copying kernels use adaptive sample lanes and cap their threads at the useful lane count. Normalization
  reuses reciprocals and combines triangular normalization with marginal accumulation. The model, search
  budgets, thresholds and stopping rules are unchanged.
- Matched local-stage wall times on a 76-core Ice Lake node fall from 373.3 to 237.1 s on K30 m1 (-36.5%),
  200.8 to 179.8 s on TroMau chr7 (-10.5%), and 120.2 to 114.3 s on std chr1 (-4.9%). N320 is within
  timing noise. These are single matched pairs, not whole-pipeline speedups; K30 uses about 7% more CPU.
- Released scientific fields agree in four matched local comparisons and their final assembly/painting checks.
  Full std passes local selection, assembly, painting, pedigree, founder refit, family phase, recombination,
  truth evaluation and all 4,727,400 records in 16 exported VCFs. Maximum local log-likelihood drift is
  2e-10; final products, evaluation and exports have zero measured drift. Kernel and parameter-split checks pass.

### Findings, no code change

- Targeted 2x/4x discovery-budget replays show mixed accuracy effects and no demonstrated downstream
  accuracy gain. Search budgets are unchanged.
- A conditional residual-seed augmentation was prototyped and rejected: final K30 genotype wrong-or-missing
  rose 1.7%, exceeding the allowed 1%. No rare-founder code change is included.
- The global tie-reversal hook changes one TroMau L2 context-assembly `release_alternatives` record.
  The isolated local winner reversal does not cause it, and released local calls remain unchanged.
  This assembly convention remains a follow-up for the assembly stage.

### Fixed

- A cold-cache crash present since 2.0: parallel kernel warm-up could leave Numba's cache index pointing
  at another overload's compiled code, causing the caller to fail to type. This occurred in a K30 simulation.
  Cached kernels are used only when their signature matches the requested one; otherwise they are recompiled.

### Validation

- Whole-pipeline runs on 76-core Ice Lake nodes agree with 2.5 in every released discovery field, every
  pedigree call, and the per-contig site counts, K, painting switches and crossovers:
  - TroMau: 9,979 blocks, 116 samples, 22 contigs;
  - the std, stress, K20, K30, f3 and reallike simulations;
  - the N-series at N = 80, 160, 320 and 640.
- Whole-pipeline wall times, 2.5 against 2.6:
  - TroMau 67.7 against 64.7 min; std 24.6 against 22.6 min;
  - N-series 10.6, 14.6, 22.6 and 45.6 against 8.2, 13.2, 20.8 and 46.1 min;
  - matched pairs run side by side: K30 46.1 against 39.0 min (-15%), K20 7.3 against 6.8 min, f3 22.7
    against 22.0 min, stress 21.7 against 21.6 min and reallike 17.4 against 18.1 min.
- A reproducer with a corrupted kernel-cache index fails on 2.5 and passes on 2.6, both in the main process
  and in pool workers. The focused tests pass.

## 2.5.0 (21 October 2026)

Release 2.5 changes the discovery stage. Discovery is practical at large sample sizes, and population-half
proposal starts are independent of any hardening convention. Released results on TroMau and the simulations are
those of 2.3/2.4 except in rare population-half blocks.

### Discovery speed at large N
- 2.3 and 2.4 had a residual-group tail. At N=320 a few blocks took tens of minutes each: chr1 of an N=320
  simulation took 2,476.9 s against 79.1 s in 2.2, and the tail made whole runs impractical. Residual alternatives
  are now counted and deduplicated as exact packed original-record memberships before Python groups are built.
  Component admission, complete size classes, omissions and every released decision are unchanged. The adverse
  N=320 block falls from 1,625.9 s and 54.4 GiB to 9.8 s and 2.1 GiB.
- Small residual pools and clique counts run inline. Larger independent pools share the live CPU budget, so
  freed cores pass to unfinished work.
- Fixed-K fitting uses guarded fractional-founder reductions. Wherever a decision could depend on them, the
  accurate-sum path decides.
- Discovery hands borrowed CPU threads to waiting block workers at its existing claim boundaries. Other stages
  keep their scheduling policy.
- Discovery wall time on the five N-series chromosomes (76-core Ice Lake nodes):
  - N=160: 7,973 s in 2.4, 235 s in 2.5 (34x faster);
  - N=320: 10,266 s in 2.4, 369 s in 2.5 (28x faster);
  - std_n80 (eight chromosomes): 367 s in 2.3, 331 s in 2.5.

### Population-half proposals
- Both coherent hardenings of population-half ties are evaluated as one proposal family, and duplicate starts
  are removed before fitting and admission. Budgets and co-winning alternatives are unchanged. Blocks without
  such ties are unchanged. On TroMau one block of 9,979 changes its panel, and the change does not reach the
  assembled founders, paintings or pedigree. Truth effects on low-coverage tests are small and mixed, and no
  accuracy gain is claimed.

### Diagnostics
- Expansion and score backstops are reported separately from ranked truncation and omitted tied classes.
- Search records persist archive-maximum K, block and phase wall and CPU times, and selected versus
  other-panel mean-field failures.
- Completed discovery blocks and their diagnostics are checkpointed every 64 blocks, or on the first completion
  at least 30 seconds after the last flush, so a stopped run keeps nearly all finished blocks.

### Removed modes
- The optional reversed-convention second search and its escalation are removed. The default search, the
  dense-geometry reference and acceptance-only reversal checks remain. Historical mode metadata still decodes.

### Findings
- Landmark (A256) and dense residual geometry agree in all 60 tested comparisons, including 20 landmark-active
  N=320 blocks.
- The TroMau chr1 and chr23 pseudo-truth regressions from 2.2 to 2.3 come mainly from long-range identity
  propagation in chromosome assembly. Block-level discovery is about equal on chr1 and better on chr23.
  Assembly is outside this release.
- Simulated K30 is under-K in 200 of 200 blocks, driven by rare founders: an information limit at N=80. K6 misses
  are near-twin founders. No tested wrong K lies within the score quantum. A broader complete-family eligibility
  policy worsened all four tested cases, so the selection objective is unchanged.
- A normalized shared-sequence panel prior was implemented exactly, including unobserved founder cells, and
  validated end to end. Gains on K30 were large: genotype wrong-or-missing -28% and founder called errors -2,911. Across seven
  held-out simulations, genotype wrong-or-missing fell 10.6%.
  But it failed the predeclared gates:
  - final founder accuracy was worse at N=80, with wrong-or-missing alleles +24%;
  - surplus-K blocks increased at N=80, at N=320 and in the backcross;
  - on K30 the called-genotype error rate rose 12% and phase switch errors rose from 192 to 272;
  - TroMau mismatches against the G0 pseudo-truth rose 5.5%.
  It is not included, and it remains the leading research item for high-K designs.
- 2.2 against 2.3 on seven paired simulations:
  - common-called genotype errors are identical in five;
  - K30 and backcross trade small genotype changes against founder recovery and phase continuity;
  - phase sets are more and shorter;
  - f3 gains four wrong parent edges.

  2.3 is therefore not more accurate overall. On TroMau, mismatches against the G0 v1.4 pseudo-truth fall
  2.8%, with chr1 and chr23 worse. These later-stage regressions are follow-ups for assembly, pedigree and
  family phase.

### Operational fix
- A cache-order-dependent Numba typing crash in family phase is fixed by passing consistent scalar types to the
  cached frame, block, cavity and polish kernels. Scientific outputs are unchanged.

### Validation
- **Scientific identity with 2.3 on TroMau:** all 22 chromosomes have identical sites, K, painting switches and
  crossovers, and no pedigree parent call changes.
- **Simulations:** std, stress and K20 show no released discovery difference from 2.3. Measured accuracy equals
  2.3 on all seven paired simulations (std, stress, K20, K30, f3, real-like and backcross). N=160 and N=320 discovery
  match 2.4 block for block.
- **Speed work is exact:** successive candidates reproduce every discovery field except timing, plus pedigree,
  K, switches and crossovers. On a three-chromosome check the exported founder and sample VCFs are byte-identical.
- **Whole-pipeline scaling (N-series, five chromosomes):** 10.6, 14.6, 22.6 and 45.6 minutes at N = 80, 160, 320
  and 640. The N=80 run includes about 2 minutes of first-use kernel compilation. The scaling exponent is about 0.8.
- **TroMau whole pipeline:** 67.7 minutes, against 72.4 minutes for 2.4.
- **Final accuracy across N (N-series):**
  - founder haplotypes are reconstructed without error at N >= 160; N=80 has 1,021 wrong-or-missing alleles per
    million;
  - genotype errors are 165-194 per million called genotypes;
  - phase switches are about 1 per million heterozygous sites;
  - there are no false parent edges.
- **Sample order:** TroMau with a permuted VCF sample order gives identical outputs.

## 2.4.0 (20 October 2026)

Release 2.4 changes the evidence stage only. Evidence products on all 22 TroMau chromosomes are identical to 2.3,
so released TroMau results are unchanged.

### Site filter: what it removes, and the heterozygote-balance rule
- The site filter keeps 33.7% of TroMau's biallelic SNPs (1,993,684 of 5,918,884). A diagnosis on chr1, chr3 and
  chr8 shows the removed sites are not clean sites lost to over-filtering:
  - At heterozygous genotypes they carry marked reference bias: the heterozygote ALT fraction is 0.38-0.43,
    against 0.49 at kept sites.
  - Their homozygote wrong-allele rate is 4-5x that of kept sites.
  - At matched depth they fit the founder paintings far worse: 0.93 against 0.06 nats per sample for
    depth-heterogeneity-only removals on chr3.

  There is no 2x excess-depth (duplication) mode. The removed sites form broad deficit-depth shoulders, consistent
  with cross-species reference divergence and mismapping. Two relaxations of the depth rules, one pooled and one
  per cell, failed a depth-matched thinning control and admitted severe high-depth inconsistencies. The filter's
  rules and decisions are unchanged.
- The heterozygote-balance rule (r4) never applies at TroMau coverage. Its clean class is exactly binomial, and
  its separation gate (three binomial SDs at the median pooled heterozygote depth) depends on coverage. A
  replacement was built and validated: a clean beta-binomial with fitted overdispersion against an artefact
  beta-binomial, separated by three intrinsic SDs. It would remove a further 2-4% of today's kept sites, and those
  sites fit the paintings about 3x worse than the sites retained.

  The replacement is deferred. On TroMau chr13 it removes the few sites that distinguish two full siblings. The
  siblings then become identical in family phase's discrete coordinate keys while their likelihoods differ, and
  family phase does not yet support such connected duplicate samples: it stops rather than order them by name.

### Discovery sites and calibration failures
- Discovery uses a fixed minimum of five total reads per site across the analysed cohort. The threshold no longer
  grows with sample count (formerly max(5, 0.02 N)), and later inference still uses the other sites. No discovery
  site or block changes on TroMau, std_n80, stress_n80, the N=320 N-series or sim_s10000.
- A read-model calibration failure no longer stops the run:
  - a shared-binomial polish failure discards only the failed start;
  - distinct successful endpoints are resolved by training likelihood only when the best leads every distinct
    competitor by more than one nat;
  - otherwise that contig uses the fixed binomial model;
  - decisions are logged ("[Read model WARNING]") and kept in the evidence diagnostics and run summary;
  - ordinary empty-fold fallbacks remain unflagged.

### Numerical record
- E3's 2.3 shared-curvature change of at most 1.97e-11 (absolute) in applied GLs, with every selected model
  unchanged, is accepted and documented.
- After the 2.1 Newton polish, GLs move at most 3.67749e-10 (relative) under a changed numerical environment across
  all 22 TroMau contigs. This replaces the stale "about 0.3 nats" limitation.

### Memory and speed
Results are unchanged.
- The VCF/BCF decoder writes straight into the final shared count array: 32.66 -> 8.25 bytes per input cell on a
  10,000-sample check.
- Input-site counts are released after site selection.
- Likelihood tables are built per active sample, and the observation mask needs no full-size temporaries.
- Record compression scratch is bounded and reserved.
- `founder_view()` shares the immutable counts and likelihoods and copies only the observation mask (1 byte per
  cell instead of 33). Local selection neutralises its private block copy, so its persisted search keys are
  unchanged.
- The site filter processes bounded site tiles with sparse sample-QC cells and reserves its real scratch. Its
  scratch is 0.54 GB at N=10,000 on a chr3-sized contig, against about 300 GB for the former whole-contig float
  arrays.
- Peaks on TroMau chr3 (N=116): site filter 4.05 -> 1.05 GiB; decode 3.08 -> 1.53 GB; read model 1.80 -> 0.49 GB;
  likelihoods 2.19 -> 0.85 GB; founder view 2.03 -> 0.92 GB.
- At N=10,000 the remaining evidence peak is dominated by the input counts and their one sample-order copy (about
  120 GB on a chr3-sized contig), down from about 0.5 TB.
- Evidence is also faster. On 76 cores the site filter with sample QC takes 8.0 s on chr3 and 9.2 s on chr7,
  against 20.0 s and 22.9 s in 2.3. Site work runs as one shared queue through the run's dynamic CPU budget,
  largest spans first, so freed cores pass to the remaining work. Mixture likelihoods are compiled and
  parallel, and fixed genotype likelihoods are cached per site.
- Decode plus site selection takes 11.6 s against 26.5 s on chr3, and 12.2 s against 28.0 s on chr7.
- Average core use in the filter is about 21%. The rest is the fits' sequential optimiser steps; in a full run
  evidence overlaps with discovery.
- Across all 22 TroMau chromosomes the largest per-chromosome evidence peak falls from 4.54 to 1.42 GiB.

### Validation
- Focused tests: 15 for the failure policy, 5 for the tiled filter, and the memory checks, including exact
  likelihood lookups and founder-view sharing.
- Evidence products identical to 2.3 on all 22 TroMau chromosomes and on std_n80 chr1/chr2.
- Downstream results identical on TroMau chr19 through discovery, local selection, assembly, painting and founder
  refit (386,368 values).
- A 3-chromosome TroMau subset completes end to end.
- No simulations were rerun: the site filter is off for simulations, and the discovery-site census shows no change.

## 2.3.0 (20 October 2026)

### Scientific model and released results

- Apply a **genome-wide consistent pedigree M1 floor**, fixed before calibration and resampling. Retain every eligible parent whose total raw M1 is at least the best eligible total minus 60 times the existing information temperature of the total markers. Retained parents keep raw chromosome scores; other identities share the exact-sum chromosome floor profile. Every eligible identity, contamination term and prior denominator remains. This changes the evidence model and can change inferred panels.
- Build M2 from **every eligible pair of retained parents**, including all cutoff ties. No top-20/40 truncation or sample-index tie break selects a prefix. The M2 likelihood model is unchanged. Excluding individually weak M1 parents can still exclude a strong raw M2 pair; this admission assumption is explicit. Missing-parent component fallbacks use the original raw M1 factors.
- Use a certified selective M1 screen by default. Every candidate whose bound cannot establish its result receives exact HMM scoring. Separate compact-profile and fixed-score checkpoints prevent a second floor transform. Four-dataset validation preserves all numerical pedigree/release fields, 4,060 bootstrap/LOCO records and 60 CSV files against the adopted exact model. Dense downstream arrays remain.
- Speed up full-marker pedigree exclusion with lower bounds from completed windows and exact completion whenever exclusion is not certified at the original genome-average threshold. Biological calls and observed-site counts are unchanged. Two nonempty diagnostic tables expose certified intervals explicitly; zero-target tables retain their original columns. Four-dataset report checks pass (TroMau/std actual G8 artifacts; stress/sim zero-target G7 carry-over). TroMau avoids 26.224% of pair-site visits: kernel CPU falls from 20.955 to 19.741 s (**5.8%**). This is a kernel gain, not a whole-stage or scaling result.
- Represent tied A-step assignments fractionally and carry complete tied discovery/selection families. Expand complete parent-score classes, rank replacement starts by their parent class, use ordinary/escalated replacement budgets of 48/96 (escalation off by default), and screen tied nearest-row/consensus merges without an index convention. Biological priors, state likelihoods and the binned switching objective remain unchanged.
- Enable 256-landmark discovery proposal geometry by default; explicit zero retains the dense reference. All observations remain in consensus, fitting and scoring, and whole over-budget classes are reported as search omissions. This is an approximation: latent panels/search paths may change even when released calls agree. Residual activation depends on the number of fractional records, not only sample count. All 38 approved accuracy cells retain their seven truth metrics; 56 invariance controls pass. These component results do not replace the final pipeline matrix.
- Finish shared-binomial read calibration at stationary endpoints with stable, content-ordered reductions. The E3 curvature and box-constrained Newton fix preserves the objective, bounds, all-start policy and standard calibration choices in 110/110 contig comparisons; the accepted maximum applied-GL residual is **1.97e-11**. Earlier stationarity changes selected three sim_s10001 models, so the broader read-model change is not described as universally output-identical.
- Family phase uses sample-name-independent coordinates, continued endpoint families and explicit display phase breaks where relative orientation is unsupported or MAP and polish disagree. These breaks preserve point genotypes and recombination selector chains; they do not claim to repair long-range orientation. The proposed **C8 orientation estimator is deferred to 2.4**.
- Integrate the exact Y sparse-support kernel and stage6ab first-pass sharing, scratch-before-CPU admission, initial work-based chromosome leases and dynamic load balancing. Eight std/stress single-contig reference/reversal/rename/noise controls each preserve all 22 released hashes and every truth/phase-set metric. Whole-stage AB timing remains separately measured.
- `haplotypes evaluate` now scores released phase sets exactly as exported. Explicit/component boundaries split phase sets; displayed-undetermined calls contribute unordered errors without splitting their phased flanks. Summaries identify `export_phase_sets_v2`. Baseline comparisons must use this same metric.
- Recombination CSR edge grouping and integer bincount totals, exact pedigree pair counting/SCCs, calibration aggregation, and compiled exact residual-clique counting preserve scientific results. Process and in-process CPU admission retain active claims until workers release them, including under memory pacing.

### Accuracy interpretation

The K30 forward-only baseline was convention-dependent. On the common set of 2,946,890 calls, 2.1 forward/reverse have 2,943/3,424 called errors and 224,959/229,273 wrong-or-missing genotypes; 2.2 has 3,159/3,396 and 226,449/230,891. Discovery candidate9c has 3,227 and 227,113: +43.5 common-call errors and -3 wrong-or-missing against the 2.1 convention mean, and -50.5/-1,557 against the 2.2 mean. These are component-stage descriptive comparisons, not proof of final-pipeline equivalence. The accepted assembly costs below are reported separately.

### Assembly and founder release

Completion's held-out folds now use exact integer total-depth vectors to distinguish
samples with the same observation mask. Identical depth groups stay whole and are
balanced deterministically between folds. The split uses no sample names, input index
or genotype-likelihood value, so sample permutation, renaming, convention reversal and
GL-only perturbations preserve fold membership. If a fold has no informative training
observations, its learned panel remains the original pre-fill panel and the absence of
held-out training is recorded. Original calls, cavity-fill and occupancy rules are
preserved. This corrects input-order-dependent completion fills.

Assembly now evaluates complete eligible join proposal classes and scores a proposed
founder panel independently of the edit used to obtain it. Tied join and count-choice
panels can continue into larger contexts, with common sample/site masks and same-level
refinement. A shared allowance of sixteen incoming/search leaves bounds higher-level
work; ordinary isolated join searches retain four. A complete encountered family is
evaluated or retained with explicit verification flags. Count candidates evaluate the
complete observed endpoint family before applying the existing acceptance rules.

Computationally incomplete count comparisons retain the incumbent decision and record
`count_unverified`, including cause, scope and domain size. They never justify a founder
deletion or withheld call. Complete nearest-shadow enumeration has a separate budget of
2048 combinations; exact conservative proofs may verify larger families. Such a proof
must hold for every sample permutation of the original fixed bootstrap draws. A
proof refusal preserves the incumbent and its verification flag. Evaluated
conflicting count verdicts retain the stated smaller-K and exact minimum-switch priors.

Founder output now records unsupported assembly continuity explicitly. Founder VCF
columns and tracks split into release segments when evaluated alternatives disagree
beyond a whole-panel row permutation. When invariant flanks establish a unique exact
row correspondence, agreeing allele cells are retained and only differing or missing
local content is withheld. For unequal-K terminal alternatives, a full-span path
occurring exactly once in every evaluated member also retains its known prepared
alleles. Unmatched duplicate rows use a conservative multiplicity rule. Unperformed continuations preserve incumbent calls and retain their evaluated
source families as `continuation_unverified`; that flag alone causes no masking.
Evaluated budget-overflow ties can still withhold their observed differing cells where
exact local provenance establishes the disagreement.

Exact bootstrap reuse and a bounded optional cache for identical refinement windows
reduce repeated work without changing scientific decisions. Cache omission and eviction
change computational work only.

#### Assembly accuracy trade-offs against 2.2

These are missing-inclusive founder allele-error deltas (point panel / released
panel), not offspring genotype or phase-error deltas. These three costs are accepted for the release. No rule or parameter was tuned to truth.

- **f3_fam: +107 / +107.** Completing the seven incoming branches under JOINT16
  lets the larger-context data choose the panel. The unique L4 join winner improves
  the join likelihood by 9.47 nats, with a 7.91-nat runner-up gap; the subsequent
  refinement objective ties. The aggregate delta comprises chr1 +108 and chr13 −1.
  This replaces a convention-dependent choice; it is an accepted accuracy cost,
  not a truth improvement.
- **K30: +26 / +26**, all on m2. Complete proposal-class evaluation replaces the
  earlier restricted winner. At the first divergence, the 73-proposal class wins
  by 0.330918 nats over the old 32-proposal class's winner (retained as runner-up).
  This is a data-decided trajectory change, not an overflow deletion.
- **Backcross: 0 / +51.** The point panel is unchanged in missing-inclusive truth;
  the released result contains 174 withheld cells and five continuity breaks.
  Withholding represents evaluated local disagreement with established correspondence.
  It is counted as missing in the reported errors, not hidden from evaluation.

Both dense simulations, std, stress, reallike6 and K20 have **0 / 0** deltas.
The dense missing-inclusive error totals are 1,108 (sim_s10000) and 37 (sim_s10001),
unchanged from 2.2. All eleven datasets' retained and withheld fractions, including
real datasets without independent founder truth, are reported in the assembly validation record.

#### Count verification and computational limits

Keep complete-shadow enumeration at **2,048 combinations** and the shared higher-level
incoming/search allowance at **16 leaves**, never 16 × 4. The corrected proof verifies
**1,501/1,549 decisions (96.9012%) on fixed14** and **1,246/1,284 (97.0405%) on TroMau22**.
These cover complete count criteria, not merely the fraction of candidate families
whose size fits the budget. This coverage is accepted for the release.

Final7's earlier 99% figures (99.0316% / 99.5327%) used a weaker, sample-order-dependent
proof. Final8 requires the stronger permutation-safe proof; it is never weakened to
meet a coverage target. On the observed domains, reaching 99% with the corrected proof
would require budgets of **16,777,216** (fixed14) and **7,077,888** (TroMau22). These are
domain projections, not runtime-validated settings, and are not adopted.

When a complete family exceeds the budget and cannot be proved, evaluate no prefix:
hold the incumbent and record `count_unverified` with cause and scope. Computational
unknowns never justify deletion or masking. Unperformed continuations similarly retain
calls and evaluated source records under `continuation_unverified`. Withholding is
reserved for established disagreement among evaluated alternatives.

### Measured resources, targets and remaining validation

- TroMau v23c4 completed on 112-core **AMD EPYC 7763**: pipeline 2,947.967 s (49.13 min), complete acceptance task 3,011.076 s (50.18 min), sampled summed process-tree RSS 62.524 GiB. The raw wall ratio to the 2.1 Sapphire reference (2,298.5 s) is 1.283; hardware differs, so this is not a software-only ratio. Summed RSS is not comparable to the historical 41.74-GiB peak-process RSS. Later exact G7/C1/G8 swaps carry biological results by their documented identity evidence, not these runtime measurements.
- The adopted G7 screen on matched 112-core EPYC simulation inputs takes 121.286 s / 3,480.03 CPU-s versus 322.982 s / 29,833.15 CPU-s exact: 2.663x wall and 8.573x CPU speedup, with 96.802% fewer exact chromosome-dyad queries.
- The whole family-phase target is about **1,200 s at N=320 on 112 cores**, originally specified for Sapphire. The latest completed Y measurement on EPYC is **1,492.322 s** for the family stage plus **321.142 s** post-fit (1,813.464 s combined), 110,610.43 contig CPU-s, 74.12 mean busy cores, 469.605-GiB sampled whole-tree peak and 83.560-GiB largest task. AB's final whole-stage measurement on the same EPYC 112-core host is **1,570.299 s** for the family stage plus 345.638 s post-fit, 110,665.76 contig CPU-s (ideal CPU/112 988.1 s), 70.47 mean busy cores, a 343.2-GiB sampled peak and 87.99-GiB largest task, with chr3/chr7 22/22 hash-identical and zero truth deltas. It was about 5% slower than Y in this run because fewer contig workers ran concurrently; the target is not met on EPYC. Available node memory and normal headroom govern concurrency, not the historical 1.5x proxy cap.
- The fixed-genome scaling target remains about **N^1.2** outside pedigree. The final N-series measures N80/N320 on the same frozen **v23c5 (G7 plus C1X)** source and EPYC hardware; N80 is 327.576 s. **N320 did not complete within its 45-minute cap:** discovery of a single chromosome took 1,564.7 s, with the search incomplete in 575 of 580 blocks, so no end-to-end exponent is measured. Discovery on this design is the open scaling issue. N160/N640 are unmeasured unless separately reported. A two-point slope is descriptive, not an asymptotic proof, and does not measure the later AB speedup.
- Pedigree's restated target is linear expensive likelihood/decision/resampling work with bounded plausible parents, a certified all-pairs bound projected at **at most 10% of whole-pipeline runtime at N=10,000**, and no dense quadratic default-path arrays beyond compact screen storage. **This target is not met:** dense downstream arrays persist. The full-genome bound cost 18.3669 ms CPU/pair projects to about 4.56 hours at N=10,000 under ideal 112-core utilization; a matched-genome whole-pipeline denominator is required before claiming a fraction. Do not divide that full-genome cost by the five-short-chromosome N-series fit.
- Final integrated truth/noise/reversal/rename coverage is incomplete. TroMau completion is not a truth or invariance pass: versus 2.2 it has 33,301 changed called genotypes, 118,698 call/unknown differences, 13,942 relative phase switches, and one changed parent set; these cross-version differences require attribution, not zero-difference acceptance. Actm3 does not carry from c3: actual residual pools reach 310, above 256, and no c4 reference has completed. Final-tree actm3 therefore rests on component evidence only. The std reference has a complete 11,822-block inactivity census (maximum residual pool 184); its reference biological outputs carry through the documented component identity chain, not through a fresh final-tree run. This does not validate its noise/reversal/renaming controls. Every deadline-skipped cell is reported as NOT_RUN, independently of pool exit status. Full final sim_s10000 truth, the remaining truth panel, broader zero loop, TroMau reversal and final Gate W remain follow-up where uncompleted; component gates remain explicitly labelled.

#### Evidence freeze and uncompleted cells

Evidence collected at **2026-10-20T08:22:56.652274+01:00**. At the evidence freeze (2026-10-20T08:22:56.652274+01:00), the exact terminal NOT_RUN list is empty: no deadline guard has yet skipped a task. Pending or running cells are uncompleted, not passes. The final report must add any later NOT_RUN records without changing this frozen transfer.

Uncompleted final truth references: stress_n80, lad_K20_n80, lad_K30_n80, f3_fam,
backcross_s6202, reallike6_nf_s3, actm3 and sim_s10000. The std reference relies on
the qualified carry-over above; its queued c4 reference has not started. All six
queued std/stress controls (iid12_s1, reversal and rename_s1 for each) are uncompleted.
N320 did not complete (discovery time cap; see above); N160/N640 are unmeasured. AB whole-stage timing: 1,570.299 s family stage (see above).
TroMau reversal, the broader zero loop and final Gate W also remain uncompleted.

### Follow-up work

Pedigree sparse downstream decisions/exposure/cavity/path/resampling, complete N-series and omitted release validations remain open. SIMD vectorisation may use the owner's post-2.3 few-ULP policy (reassociated reductions/FMA only when released decisions are unchanged); this policy is not applied retroactively to 2.3's exact-swap gates. C8 orientation estimation continues for 2.4. G8 is included after its exact-exclusion, interval-report and default-handler/checkpoint gates; its modest kernel saving does not resolve the remaining pedigree scaling requirement.

## 2.2.0 (19 October 2026)

Release 2.2 removes arbitrary conventions from every decision that selects between different biological answers.
Tied alternatives that more data can separate are carried forward and decided by the data. Ambiguity the data cannot
resolve is released as such: unknown alleles, switch intervals, phase-set breaks, unordered parent slots, and flagged
alternative sets. The remaining exact ties are decided by stated priors ([docs/methods/ties.md](docs/methods/ties.md)).

Validation of the release candidate (19 October):

- **Noise loop.** Released outputs were identical in every cell. Compared were founder alleles (unknowns
  included), founder counts, painting (intervals included), pedigree, phased calls, crossovers and the VCFs. The
  cells were 1e-12 and 1e-9 relative genotype-likelihood noise, value-consistent noise, permuted sample order (in
  memory and in the input file), another thread count and another numerical environment. They covered three
  TroMau chromosomes, std_n80 and stress_n80, and the TroMau whole genome under 1e-9 noise and permuted order.
- **Convention reversal.** Released outputs change only through the conventions listed in
  [docs/methods/ties.md](docs/methods/ties.md#remaining-conventions):
  - stress_n80: no genotype changes; 13 phase switches; crossover intervals of 26 meioses.
  - std_n80: 6 genotypes; 20 phase switches; 20 meioses.
  - Three TroMau chromosomes: 1,361 genotypes and 1,713 phase switches, from 26 discovery blocks.
  - TroMau whole genome: 22,827 genotypes and 11,056 phase switches, from 326 of 9,979 discovery blocks (2.1
    with the reversal switches: 46,758 and 20,021). One Tier-B status changes.
- **Truth against 2.1.**
  - Pedigree calls are identical on every simulation evaluated, with one more exact configuration on the K = 30
    ladder. sim_s10000 was still running at release.
  - stress_n80: genotype errors -20%; allele errors per released phase set -39%, with 137 more phase sets (N50
    -0.6%); phase switch errors 98 -> 94.
  - std_n80: allele errors per phase set +19%, all from one sample's orientation at a painting tie boundary;
    genotype errors -148.
  - f3_fam: allele errors -6.6%; genotype errors +7%.
  - reallike6: allele errors -8%.
  - backcross: allele errors per phase set -9%, with 33% more phase sets; genotype errors +0.3%.
  - sim_s10001: genotype errors +4 of 2.87 billion calls; allele errors per phase set -15%, with 7,040 -> 11,168
    phase sets (N50 -5%); 3.3% of called alleles are released unphased where the frame is undetermined.
  - K = 20 ladder: identical.
  - K = 30 ladder: genotype errors +9% and phase switch errors 170 -> 216, mostly the cost of the order-free
    A-step rule. The final panels recover the same 143 of 150 true founders as 2.0 and 2.1.
  - On a metric that splits each sample at the painting tie boundaries of both releases, stress_n80 is 19% worse.
    That comes from long orientation errors in four two-parent samples; f3_fam and reallike6 are better on it.
- **Resources.** The TroMau whole genome took 37:12 against 2.1's 38:23, with peak memory 42.3 GiB against
  41.7 GiB. Dense simulations take up to 1.13x 2.1's wall time and 1.06x its memory.

### Declared model changes

- **Discovery: tie-carrying residual clustering.** Every tie resolution of the residual clustering is carried as a
  candidate for the block score to decide: level-wise HDBSCAN* condensed-tree groups and maximal-clique alternatives
  of the noise, with order-invariant bounds. It replaces 2.1's content-key order of equal-weight links and the
  complete linkage on noise. Archive beams keep whole NLL classes. Emission, deduplication, K = 1 seeds and the
  replacement and gauge budgets are order-invariant. Founder alleles that no carrier observes are released unknown
  (`latent_unknown`). NLL-tied panels are released with the block (`tied_panels`).
- **Discovery: tied A-step states take their meet.** A sample whose tie band holds several states (within the
  fewest-unknown-copies category, a kept prior) is assigned the multiset intersection of the founder pairs; a
  copy that differs between the tied states goes to no founder. This depends only on the tied set, not on row or
  sample order, and replaces 2.1's row-order rule in discovery. The objective is unchanged: the unknown-copy
  count stays the tied states', and the cavity's state prior counts a tied sample 1/|T| at each tied state.
  Accepted cost on the K = 30 ladder: discovery misses 6 more true founders, summed over blocks, than with the
  row-order rule (10 fewer than 2.0, 3 more than 2.1). After assembly every release recovers the same 143 of 150
  true founders, but genotype errors rise by 6% and phase switch errors from 170 to 216. A fractional treatment
  is the planned next step.
- **Local selection: tied sets decided by the data.**
  - Every source's cavity tie set is fitted and searched, instead of 2.1's last-source start.
  - The BIC tie set of the bank is kept whole, and tied donor completions each give a source
    (`CandidateBankConfig.completion_variant_cap` = 8).
  - Whole tie classes are refitted (`PathSearchConfig.refits_per_kind_cap` = 16).
  - Residual ties give a consensus panel that withholds rows not shared by every alternative (`tied_alternatives`,
    `withheld_by_ties`).
  - Discovery's tied panels become starts and bank members.
- **Assembly.**
  - Batching is orientation-symmetric.
  - Inside log-likelihood tie classes the batch linker's path score decides join moves.
  - Painting-dependent proposals run under both traceback conventions and are merged.
  - Tied count-step refits that disagree on the founder count take the smaller count (prior); otherwise the member
    whose margin most strongly supports the decision wins.
  - The two fixed-K refinement searches use opposite conventions and keep the better F. Equal-F alternatives are
    flagged (`ambiguous_k`, `alternatives`).
  - Founder refit releases mutually exclusive tied moves as unknown cells.
- **Painting: switch intervals.** A switch between founders the sample cannot tell apart is released as an interval
  (`SWITCH_INTERVAL`) instead of a point at the leftmost tied edge. Tracks next to an interval are released only when
  2.1's gate passes under both traceback conventions, which is never more permissive than 2.1. Founder alleles inside
  interval bins are those both candidates share. On stress_n80 this removes 366k calls, of which 6.4% were wrong
  (genotype errors 117,369 -> 93,909), at a cost of 11 more phase switch errors (98 -> 109). The per-sample segments
  are published (`painting/<contig>/sample_segments.tsv`).
- **Pedigree: parent slots.** Parent1/Parent2 follow recorded sexes (dam, sire). Without recorded sexes the slots are
  unordered (`ParentSlotBasis`).
- **Family phase: a schedule-independent family MAP and released undetermined frames.**
  - The ascent runs under the canonical schedule and under its mirror. Difference clusters are transplanted,
    best exact gain first, into the end point with the higher posterior, and the full ascent then continues. It
    stays monotone with exact gains.
  - New moves:
    - compound moves with terminal segments;
    - a joint sibship genotype refit inside compound seeds (at most 64 joint configurations per site, whole sites
      skipped, never truncated);
    - sibship pairs with tolerance;
    - co-located pairs (siblings switching different parents' bits at the same markers).
  - The long compound tier is off by default (`FamilyPhaseConfig.map_long_moves`).
  - `frame_undetermined` marks sites where another tied configuration sets the frame differently (bit 0
    decoder, bit 1 displayed), and the export breaks the phase set there (`phase_breaks`).
  - One-parent children are shown in parental origin: the painting phase inside each stretch between its painting
    tie boundaries, each stretch oriented by its summed family-MAP evidence.
  - Root-frame repair votes: a child opposes a change only if its exact gains agree with its flanks. This removes
    both 2.1 transfers on stress_n80, including chr10 F1_8 (+43k allele errors in 2.1).
  - Recombination decodes the non-homozygous sites of an undetermined decoder frame as unknown, so crossovers do
    not depend on where a tied switch is placed.
- **Export.** Founder alleles with no observing painted carrier are written `.`, counted in the manifest and the
  founder VCF header (`unobserved_founder_alleles`). Phase sets break where family phase leaves the displayed
  orientation undetermined.

### Convention reversal and evaluation

- `HAPLOTYPES_TIE_REVERSE=1` reverses every remaining arbitrary convention: index, content-key, default-state,
  sample-name and keep-first orders (`maths.ties.reverse_conventions`). It is the release acceptance test.
  `HAPLOTYPES_TIE_REVERSE_KEEP` keeps named discovery conventions at their default for attribution. Records written
  under reversal carry `tie_reverse` in their identities.
- The evaluation reports allele errors aligned per released phase set (`ps_aligned_allele_errors`), the number of
  phase sets with at least two called heterozygotes (`phase_sets`), their N50 (`phase_set_n50_bp`) and the called
  alleles of undetermined frames. The component-aligned metric is kept for continuity, but it charges a released
  phase-set break as orientation errors.
- The run report gains `painting/<contig>/sample_segments.tsv`, and `ties/summary.json` records whether conventions
  were reversed.

## 2.1.0 (19 October 2026)

Release 2.1 makes the results independent of floating-point rounding, of the input sample order, of the thread count
and of arbitrary tie-breaks, replaces family phase's loopy belief propagation with a convergent solver, and searches
discovery more broadly. On the release candidate of 18 October, the noise-stability matrix found zero changed outcomes
at every stage, from evidence to export. The matrix covered 1e-12 and 1e-9 relative genotype-likelihood noise,
value-consistent noise, permuted sample order (in memory and in the input file), a changed numerical environment
and a different thread count, on simulated crosses and the TroMau benchmark. Against 2.0, that candidate gave:

- identical pedigrees on every simulation, and one more correct edge on the K = 30 ladder;
- phase switch errors and component-aligned allele errors within a few percent either way (allele errors -22% to
  +6.5% by simulation);
- on the TroMau benchmark, slightly lower founder coverage (best-row coverage -0.012) and fewer painted F2
  switches;
- TroMau wall time within +6% of 2.0, with a peak memory of 62 GB against 44-59 GB.

The release adds the family-phase solver and late fixes to that candidate. It was checked end to end before
release: three TroMau chromosomes and the std_n80 and stress_n80 simulations completed cleanly. Their pedigree calls
equal the candidate's, and so do std_n80's genotype, phase-switch and allele errors. On stress_n80, allele errors
are +1.6% from the new family-phase solver. The full noise matrix was not rerun on the final code.

### Declared model changes

- **Local selection, context code: claims coded by the cheapest matching.** Context founders claim panel rows by a
  maximum-cardinality, then minimum-edit-cost matching of founders to their nearest rows (the panel's shortest
  description), so the code length is a function of the panel's rows whatever their order. 2.0's greedy rule (a
  founder equidistant to several rows claimed the lowest; applied to a trajectory-dependent row order, it let the
  same row set cost up to ~140 nats more or less) is removed.
- **Founder refinement: zero-gain founders are deleted first.** A founder whose removal leaves every sample's score
  unchanged (within the tie quantum) is removed before each deletion round, which can lower the founder count where 2.0
  kept a founder no sample uses.
- **Discovery: broader search.** More data-birth attempts and children (x3), beam width 6, and tie groups of
  equally ranked data births consumed whole. The founder-count criterion is unchanged; the broader search finds
  better-optimised lower-K panels more often, which lowers K in some blocks (accepted).
- **Family phase: loopy belief propagation is replaced by a provably convergent posterior-mode solver** (the family
  MAP). It runs monotone exact block coordinate ascent, with pedigree roots' genotypes summed out. Every accepted
  step raises the posterior by more than a tie quantum, so the solver terminates; reaching the sweep cap
  (`FamilyPhaseConfig.map_max_sweeps`) is an error. Decoder frames, root-frame repairs and crossovers can change
  where 2.0's propagation had not converged. Known limitation: the solver can stop at a local optimum; larger
  moves are planned for 2.2. `family_phase/summary.csv` reports `map_sweeps`, `converged` and `map_objective`
  instead of `iterations`, `converged` and `final_residual`.
- **Read model: fits polished to their exact optimum**, so genotype likelihoods do not depend on the numerical
  environment (BLAS kernels, SIMD paths).

### Noise stability

- One canonical sample order per run (samples sorted by name) from the evidence stage on; exported VCF columns and
  per-sample report tables keep the input order.
- Tie classes with per-kind tie quanta (`maths/ties.py`) replace exact floating-point comparisons and array-order
  tie-breaks in discovery, local selection, assembly, painting, pedigree, family phase and recombination; ties are
  decided by exact content keys and every panel is kept in canonical row order. Threshold comparisons get a band
  only where a value can sit exactly on the threshold mathematically.
- Results no longer depend on the CPU's vector instructions. Discovery's HDBSCAN (K = 1 seeds, residual-strand
  clusters) orders links of equal weight by a fixed pseudo-random key of the canonical record indices; the hdbscan
  package's unstable sort of equal-weight edges differed between AVX-512 and other CPUs. This equal-weight-link
  rule is a content-key convention under review. Logarithms and exponentials in discovery and the block scale go
  through scalar libm, and Numba's Intel SVML is disabled (`NUMBA_DISABLE_INTEL_SVML=1` at package import).
- Local selection decides ties without index or hash order. Bank winners inside wide tie classes are ordered by
  (K, canonical key). Proposals tied with the eighth refit of a kind are refitted too, up to
  `PathSearchConfig.refits_per_kind_cap` = 12. A larger MAP unknown-copy count must win by more than
  `PathReleaseConfig.unknown_count_band` (2^-20) of the largest count mass, else the smaller count stands.
- Founder refinement's exchange and content moves, its coarse-move guard and founder refit's interacting allele
  moves are decided by tie classes of their exact gains, then content. Founder refit judges each site's moves
  against the round-start state; it is identical to 2.0 when candidate moves do not interact.
- Ties are decided by documented deterministic conventions and counted. Where alternatives tie (exactly equal
  support, or within one tie quantum) a fixed rule decides: content keys, a parsimony preference (smaller panel,
  fewer parents, fewest switches), keeping the current state, sample-name order, a conservative threshold side,
  or abstention. [docs/methods/ties.md](docs/methods/ties.md) lists them per stage. They are conventions, not
  inferences, and are under review for 2.2. A completed run writes `ties/decisions.csv` and `ties/summary.json`
  (decisions taken inside a tie per contig, stage and decision kind; discovery blocks selected inside a tie,
  their tied alternatives and score margins; ambiguous founder-count calls; tied refits that disagree) and a
  one-line digest in the log; the fields they come from (`tie_events`, `ambiguous`, `tied_alternatives`,
  `score_margin`) are documented there. The plan for 2.2 is to carry every tie resolution forward as a
  candidate and let the block score decide.
- Fixed: an assembly join could cycle forever between a birth, a death and a seed swap (count-state revisit guard
  and a reported backstop).
- New diagnostic environment variable `HAPLOTYPES_TIE_CENSUS` (valley census of tie decisions; results do not
  depend on it).

## 2.0.0 (unreleased)

The package is rebuilt from release 1.0 (commit 92b3daf) as one pipeline of typed stage packages joined through the
checkpoint store, with one driver, one configuration and one execution layer. The science is that of 1.0: the
algorithms, models, thresholds, release and tie rules are unchanged, apart from the fixes below. Accepted on
18 October 2026 against 1.0 in the same environment on nine reference scenarios
([validation](docs/validation.md#results)): every difference is attributed to a documented fix or to rounding and
ties, the pedigree truth metrics are identical on all seven simulations, the whole-genome simulations run 3-5% faster
with a 27-34% smaller largest process, and TroMau takes the same wall time (41:10 against 41:00) with a 16% lower
node peak. [Migration](docs/migration.md) maps every old command, setting and output name to the new one.

### Incompatible changes

- One command, `haplotypes run`, replaces `simulate`, `reconstruct`, `tropheops` and `astcal`; the crosses are the
  presets `tromau` and `aulstu`. `run.py` is removed. `evaluate`, `export`, `status`, `example`, `recombination` and
  `batch-worker` take run directories and destinations as positional arguments.
- The configuration has new sections (`[execution]`, `[input]`, `[samples]`, `[quality]`, `[genetic_map]`,
  `[pedigree.eligibility]`, `[family_phase]`, `[simulation.reads]`, `[reports]`). Unknown keys and sections are errors
  (1.0 ignored them). Scientific switches are `--set SECTION.KEY=VALUE` instead of one flag each. `[input] contigs` is
  required for VCF input unless a preset lists them.
- No setting is read from environment variables any more (`HAPLOTYPES_*`, `BHD_*`, `HAPREC_COARSE_MOVE_GUARD`).
- One stage vocabulary: `block_discovery` is `discovery`, the feedback rounds are `local_selection`
  (`local_selection_initial`, `_l1`, `_l2` in evaluation), `allele_refit` is `founder_refit`, family refinement is
  `family_phase`. Output directories are named after the stages (`quality/`, `painting/`, `founder_refit/`,
  `family_phase/`, `recombination/`, `simulation/`); `pedigree/scientific_relationships.csv` (a copy of
  `tier_b_relationships.csv`) and constant diagnostics columns are removed.
- Checkpoints written by 1.0 are not read: the container format is unchanged, but record keys and identities are new.
  A stage's identity covers the source of the packages it runs (computed automatically), its configuration and the
  record identities of its inputs.
- Removed alternative modes: founder refinement always runs (`--founder-refinement off` is gone) and its coarse-move
  guard is fixed to the shared-junction guard; the pedigree direction alternatives (cluster, continuous-only and
  family-only) and their internal ablation switches are gone; the pedigree keeps one current record per product
  instead of versioned decision replays, so a changed pedigree setting needs a new output root.
- Python callers: `core/` and `workflows/` are replaced by `config/`, `store/`, `execution/`, `pipeline/` and one
  package per stage (`evidence/`, `discovery/`, `local_selection/`, `assembly/`, `painting/`, `pedigree/`,
  `founder_refit/`, `family_phase/`, `recombination/`), with `evaluation/`, `reports/`, `export/` and `templates/`.

### Input and checks

- One record policy for every VCF/BCF: records that are not biallelic SNPs are dropped and counted (`not_snp`,
  `multiallelic`), every record at a position with two or more biallelic SNP records is dropped
  (`duplicate_position`), and I/O or format errors stop the run. Real BCFs with indels are accepted. Results are
  unchanged wherever 1.0's site filter ran (its rule r0 removed the same records); 1.0's cross route truncated
  multiallelic depths to the first ALT and could swallow read errors, its generic route rejected any file with an
  indel, and runs without the site filter read indels as SNPs.
- Everything that can be checked runs before any computation: configuration consistency, the variant header (AD,
  samples, contigs present and populated), the workbook, sheet, columns and sample list, and the eligibility policy
  (1.0 read the cross workbooks only after painting). Runs reaching the genome-wide stages need at least three
  contigs; a run with fewer than five contigs warns at the start that Tier-B calls are unreachable, and the pedigree
  states its count of informative chromosomes and whether each tier was reachable.
- Genotype likelihoods of every read model come from one emission kernel; values change at rounding level only.
- Read model: the shared homozygote-overdispersion candidate and its null refit are removed (homozygote kernels are
  binomial); held-out selection scores within one nat of the best are ties and the simplest candidate is applied;
  optimiser stop reasons do not decide which fit is used (heldout-read-model-v6).
- A run marks its checkpoint directory with the release layout before writing anything and refuses a directory
  holding checkpoints without the mark (an earlier release) or with other subdirectories. A shard of a simulation is
  refused until the simulated inputs of every contig are recorded.

### Scientific fixes

Each was measured on the bounded references against the unfixed tree; [validation](docs/validation.md) has the
acceptance evidence.

- Discovery: a cycling fixed-K fit returns the cycle's lowest-NLL state, reports the iteration limit and is never
  certified as a fixed point (1.0 returned the state the iteration limit landed on, which depended on the limit's
  parity). Cyclic-fit endpoints changed; no block outcome changed on the references.
- Assembly count rule: the row geometry (nearest rows, shadows, near-identity and novelty tests) reads the pre-fill
  calls, so a completion fill never decides a count (1.0 measured on the published calls). Founder refinement's count
  rule counts S as the component's informative samples (1.0 counted every evidence row, founder-excluded samples
  included), and a deletion round is skipped by its bound only when the bound also proves the bootstrap band (1.0
  recorded such rounds as not ambiguous without evaluating the band).
- Painting: bin-to-bin switch probabilities use Haldane's map function (1.0: min(1/2, d)); release status changed in
  a handful of track bins with posteriors within 1e-3 of the 0.9 threshold.
- Founder refit: two-row moves are only swaps of different alleles of rows painted together at the site, accepted
  at twice the per-allele prior (10 nats); 1.0 flipped any pair of called rows on a summed 5 nats.
- Family phase: the polish never switches phase inside the first painting bin of a chromosome or component (1.0 gave
  those intervals the switch probability of a bin edge).
- Pedigree: the weakest-link tie tolerance of acyclic graph selection is 512 machine epsilons of the scores the
  margins come from (1.0 took 512 epsilons of the margins themselves, below their rounding).

### Fixes and other changes

- New export stage: every complete run writes a founder VCF and the phased sample VCF per contig to
  `OUTPUT/export/` (`[export] enabled`, `[export] full_length`); `haplotypes export` writes the same files, with
  per-component founder-track TSVs as the optional product `tracks`.
- New memory governor: parallel work is admitted by estimated peak bytes and waits when memory is tight instead of
  failing; `[execution] memory_limit` (`--memory-limit`) caps it, nested work keeps headroom for its sub-tasks, and a
  new run removes the shared-memory segments of dead runs. Pacing never changes results.
- Large founder counts: local selection, discovery, assembly and painting keep their working memory per task and
  reserve calibrated estimates (lane-block path kernels, bit-packed discovery caches, shared assembly evidence,
  batched label continuity); results are identical. Founder refinement's window programmes run faster with identical
  results.
- One CPU-lease tree and one process-pool type replace the separate redistribution mechanisms; the Numba threading
  layer is OpenMP explicitly (the `tbb` dependency is dropped).
- The resolved configuration is written to `OUTPUT/config.toml` and into every run record; stage timings include
  helper threads.
- Discovery's checkpoint chunk counter advances only after a successful write, so a failed write can no longer make
  an incomplete contig look complete. Maximum cuts above `DiscoveryConfig.search.exact_cut_max_k = 12` founders are
  reported as heuristic cuts instead of a search limit (diagnostics only).
- Assembly levels continue past L4 while a boundary that is not a phase break remains (no change at the reference
  sizes); final and context assemblies are distinguished explicitly; joined batches are keyed by their typed inputs,
  so identical joins are reused across a contig's assemblies more often.
- Founder refinement's window frontier has a fixed cap (`AssemblyConfig.refinement.frontier_bytes`, 128 GiB) with
  each fallback reported, instead of an error above a fixed entry count; sweep histories are stored once per fit.
- The painting stage owns its configuration and release rule, computed once when it is written; its memory budget
  reads the job's memory (cgroup-aware) rather than the node's.
- Pedigree identities are per contig, the evidence calibration comes from the configuration, memory planning reads
  the records instead of a published table, and the two roles of the former parent-state contamination constant are
  separate fields (same values). `OUTPUT/pedigree/summary.json` publishes the informative contigs, the
  chromosome-count statement (whether Tier A and Tier B were reachable), the family solve and the exclusion and
  eligibility summaries.
- Founder refit binds its identity before reusing a recorded contig, as every other stage does.
- Recombination's shared-orientation fit no longer fails on genetic maps with zero-rate stretches.
- Family phase releases after at most 25 belief-propagation iterations (earlier when converged). 1.0's stability
  checks after 20 iterations could not fail under the scaffold polish, so outputs are unchanged; the retry ladder,
  the 520-iteration limit and the switches restoring older behaviour are gone. Protected intervals
  (`[family_phase] protected_intervals`) and order breakpoints share one reader that requires the header line; the
  tables' content, not their path, enters the identities.
- The reference-consistency report uses the run's calibrated genotype likelihoods (descriptive report only).
- `painting/<contig>/founder_counts.tsv` and `assembly_search.tsv` count each join search once, at the level where
  it ran; the genome-wide `recombination/junctions.csv` is built from the records at full precision; the simulation
  truth pedigree table is also written on resume (columns `sample`, `cohort`, `parent1`, `parent2`).
- Export takes the biallelic SNP record at positions with several records.
