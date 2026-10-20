# Family phase

Family phase (stage `family_phase`, entry point `family_phase.stage.run`) gives every sample a final,
pedigree-conditioned phase of every chromosome while preserving its called genotypes and missingness. It starts from
a phase scaffold decoded from the [painting](painting.md) and the [founder refit](founder_refit.md), computes the
posterior mode (the family MAP, pedigree roots' genotypes summed out) of a family model conditioned on the
supported Tier-B parent edges released by [pedigree inference](pedigree.md) by monotone exact block coordinate
ascent, repairs pedigree roots' scaffold frames
by the votes of their children, and polishes the
frames of children with two observed parents by a genotype-preserving copy-consistency search. It runs genome-wide
after the founder refit, one chromosome per worker; [recombination](recombination.md) decodes its product. Family
phase does not modify the pedigree, the painting or any founder panel, and nothing feeds back upstream.

## Inputs and outputs

| | record or setting | content |
| --- | --- | --- |
| reads | `painting` | per component: bin edges, eligibility, released candidate rows (`member_ids`, `member_sets`), local allele keys, trajectory classes, released chunks (no panels, no emissions) |
| reads | `founder_refit` | the refit panels' inference alleles (refit alleles on the frozen snapshot's called cells) |
| reads | `evidence` | normalised genotype likelihoods `float64[N, L, 3]` and the positive-depth mask of all samples, every evidence site |
| reads | `ParentEdges` | the supported Tier-B parent → child edges (parent, child, slot 0/1 for the child's Parent1/Parent2 column) |
| reads | the contig's genetic map | meiosis switch probabilities (Haldane) |
| reads | `[family_phase] protected_intervals` | optional breakpoint TSV of intervals where no root frame is repaired |
| writes | `identity/family_phase` | `FamilyPhaseConfig` (protected intervals included), the identities of `painting`, `founder_refit` and `evidence`, the edges' identity, the map's digest, code identity |
| writes | `family_phase` | the `FamilyPhaseProduct`; run record `family_phase` (summaries) and `complete/family_phase` |
| publishes | `OUTPUT/family_phase/summary.csv` | one row per contig |

The other `FamilyPhaseConfig` fields are internal model constants ([configuration reference](../configuration.md)).
Before phasing a contig the stage checks that its painting was computed from the current evidence record and that
the pedigree and evidence sample axes agree.

## Phase scaffold

The scaffold (`family_phase/scaffold.py`) gives every sample two track alleles at every evidence site of the
chromosome. Inside each component's painting bins, a track's allele is that of its released candidate rows in the
refit inference panel: the row of a named track, or the common allele of a pooled track's rows where every member
is called and they agree; unknown (−1) otherwise, never argmaxed to REF. Abstaining tracks, ineligible samples,
sites of a component's span missing from its panel and sites between components are unknown. The scaffold's track
order is the painting's (gauge-fixed by its tie rule) and is called the scaffold orientation.

- **Phase bins** are the painting bins, so frames are defined per bin edge. A component without painted sites uses
  bins of `family_phase.scaffold.EMPTY_COMPONENT_BIN_SITES = 100` panel positions; it carries no scaffold alleles
  and no coverage.
- **Coverage.** A sample's coverage is the bp span of every component in which it was painted (eligible),
  abstentions included; recombination decodes meioses only inside coverage.
- **Tie boundaries.** For parents with at least `root_frame_minimum_children` children, the chunk boundaries where
  the painting's track order was set by a tie (the ordered and the swapped continuation change equally many
  candidate rows and equally many allele-visible keys) are marked; frame repair treats them as places where the
  root's own data do not carry its orientation.

## Family model

The variables of one chromosome (`family_phase/solver.py`, `family_phase/potentials.py`,
`family_phase/factors.py`):

- **Ordered genotypes**, four states per sample and site (2 × first-track allele + second-track allele). Genotype
  potentials use the normalised likelihoods at observed sites and 1/3 elsewhere (probabilities floored at 1e-30),
  conditioned on the scaffold: two called scaffold alleles fix the dosage, one called allele allows that allele
  plus 0 or 1, and impossible ordered genotypes get probability 0. Pedigree roots (no observed parent) add the
  prior (1/3, 1/6, 1/6, 1/3) on (0|0, 0|1, 1|0, 1|1). The genotype of every site with a called scaffold pair is
  therefore fixed; only its order is inferred.
- **Phase chains**, one per pedigree root with a scaffold-heterozygous site: a binary chain saying whether the
  root's frame matches its scaffold orientation, acting on its two heterozygous ordered states with error
  `FamilyPhaseConfig.scaffold_phase_error = 0.01`. The frame may switch only at painting-bin edges, with probability
  `FamilyPhaseConfig.phase_switch_probability = 0.005`, and with probability 1/2 at component edges; never inside a
  bin. A root with exactly one child edge keeps its frame inside each component (its chain switches only at
  component edges): with one child, a frame change of the root and a crossover of that child explain the data
  equally well. The chain is anchored at the root's first scaffold-heterozygous site, which fixes its gauge.
  Runs of zero switch probability are contracted exactly, so each chain runs over painting-bin runs, not sites.
- **Copy factors** join each child with at least one observed parent to those parents through two transmission
  selectors, one bit per parental slot saying which parental homologue the child's track in that slot copies; each
  copy errs with probability `FamilyPhaseConfig.transmission_error = 0.001`. An unobserved parental slot transmits
  either allele with probability 1/2.
- **Selector chains**, one joint four-state chain per child (its two selector bits), with Haldane switch
  probabilities (1 − e^(−2d))/2 of each marker interval from the genetic map. The chains run continuously across
  assembly breaks (meiosis is continuous there; only frames reset). Markers where every observed parent is a hard
  homozygote carry no selector evidence and are integrated out exactly through the composed switch probability
  (1 − Π(1 − 2θ))/2.

A hard homozygote has a single possible ordered genotype, so every maximisation below sees one state there.

## Family MAP solver and release

The released estimate is the family MAP: the mode of the posterior of the frame chains F, the selector chains S
and the ordered genotypes G of every sample with an observed parent, with every pedigree root's genotypes summed out
site by site (the Lander-Green treatment of founder alleles). Its log is

    J = Σ_children Σ_sites potential(G) + Σ_children [log 1/4 + selector chain(S)]
        + Σ_roots [log 1/2 + frame chain(F) + Σ_sites R] + Σ_children Σ_sites log copy (observed non-root parents)
        + log 1/2 per unobserved parental slot and site,
    R = log Σ_g exp( potential_root(g) + log φ(g | F) + Σ_children of the root log copy(child's allele | g, S) ),

where φ is 1 − 0.01 for the heterozygote the frame matches, 0.01 for the other and 1/2 for homozygotes and
unoriented sites, and copy is 1 − 0.001 per transmitted and 0.001 per mistransmitted allele. Summing a root's
genotype keeps the evidence of allele sharing between its children where the root's own scaffold has fewer than two
called alleles: a maximised root genotype would explain shared alleles by a homozygote and discard it. Given the
children's selectors and genotypes the copy factors factorise over the slots, so each R is an exact four-term sum.

`family_phase/map_solver.py` maximises J by monotone exact block coordinate ascent (release 2.1). Releases up to 2.0
ran damped loopy belief propagation, which has no convergence guarantee on this graph of short loops (along the
chromosome and through multi-parent families) and cycled where a family's data conflict, so that a capped release
depended on the phase of the cycle.

- **Initialisation** (one top-down sum-product pass): every frame 0 (the scaffold orientation); child blocks in
  generation order, each set to its exact optimum given its parents' genotypes summed: a root's potential, φ and
  its already initialised children's factors; a non-root parent's stored conditional genotype distribution given
  its own selector path, with its already initialised children. Tied genotype orders of a parent therefore carry no
  phase evidence to its children (a maximised parent would resolve them to 0|1).
- **Child block** (a child's selector chain and genotypes, everything else fixed): Viterbi over the child's
  supported markers with, for selector state s at a site, the maximum over the child's ordered genotype of its
  potential, its slot terms under s and its own children's copy factors. A root parent's slot term is that root's R
  with the child's factor in (the root's cavity without the child, its other children in), so siblings conflict
  through their parent's summed genotype; an unobserved slot's term is 1/2.
- **Root block** (a root's frame chain): Viterbi over the root's frame-chain runs with, for frame f, the sum of R
  over the run's sites; the anchor run keeps frame 0.
- **Relabelling moves**: for root r and every boundary of its frame chain (component edges included), r's homologue
  labels are flipped from the boundary to the chromosome end, re-gauged at its anchor (the side without the anchor
  is flipped): r's frame and r's bit in every child's selector. R and every other term are symmetric under it, so
  the change of J is r's frame transition plus each child's selector transition across the boundary. A root with a
  single child moves only at component edges.
- **Compound moves** (exact, each accepted as a whole on its change of J; candidates are a selector's change pairs of
  any span and, since release 2.2, the terminal segments between a chromosome end and the nearest change, which the
  move then removes):
  - a child's selector change pair is removed and its genotypes refitted, then its
    parents (root frames, non-root blocks), their other children, its own children and itself are re-decoded exactly
    inside a window around the pair;
  - a non-root parent's two tracks are swapped on such an interval (its own change pairs and each child's change
    pairs of the bit for it) together with every child's bit for it, which leaves the children's copy factors
    unchanged; then the same windowed re-decodes;
  - the children of a root whose change pairs of their bit for it lie near each other (each end within the pair's
    span; release 2.2, either homologue) change back together, each on its own pair, then the same windowed
    re-decodes;
  - co-located pairs (release 2.2): every bit of the child and of its parents' other children whose change pair lies
    near the child's pair, whichever parent the bit belongs to (full and half siblings that all switch their
    non-shared parent's bit at the same markers, where the data say one sibling switched the shared parent's bit),
    change back together, then the same windowed re-decodes (pairs of at most `COMPOUND_WIDTH` sites);
  - a compound relabelling: a root is relabelled at the frame-chain boundary nearest a child's change of its bit
    for it (within `family_phase.map_solver.RELABEL_REACH = 4096` sites), then its children and frame are re-decoded
    inside the window spanning both: a child's switch that sat elsewhere in the stretch where the root's bit is
    uninformative for it moves to the boundary or vanishes. This is the move for a scaffold frame error that the
    children's switches do
    not meet exactly.
  After the seed, the genotypes of the sibships involved (the child's parents' children, and the child's own when it
  is a parent) are refitted JOINTLY at each site of the seed interval (release 2.2; at every site where at least two
  members have more than one possible ordered genotype for seeds of at most `COMPOUND_WIDTH` sites, at
  Mendelian-conflict sites for longer ones; the site-move search below, with at most
  `family_phase.map_solver.JOINT_REFIT_COMBOS = 64` joint configurations per site except at the conflict sites of
  short seeds, `SITE_MAX_COMBOS`; a site with more is skipped whole, never truncated), because sibling genotypes
  coupled through a root's summed genotype often have to change together. Windows widen the interval by max(64, min(span, 1024)) sites
  on each side (for intervals longer than four times that reach, only the two zones around the interval's ends are
  re-decoded; its interior keeps the seed's state); the windowed re-decodes pin the states outside the window, and a
  move's gain is the change of the J terms it can alter, evaluated by one routine before and after.
- **Site moves** (sibship genotypes): at every Mendelian-conflict site of a root with at least two children or of a
  non-root parent, the exact joint maximum over the ordered genotypes of its children (and of the parent itself
  when it is not a root), selectors fixed; candidates are the genotypes of finite potential (at most
  `family_phase.map_solver.SITE_MAX_COMBOS = 4096` joint configurations). Full siblings that must swap together which
  parent transmitted which allele, or several children whose genotypes must change together to remove a transmission
  error, need it: no single child block can make such a change.
- **Acceptance and schedule.** A block's new path is split into the maximal intervals where it differs from the
  current configuration (intervals joined by a transition are merged); each interval, relabelling or compound move is
  accepted only if its exact gain (one scoring routine before and after, never a dynamic-programming maximum)
  exceeds `family_phase.map_solver.GAIN_QUANTUM` (2^-17 nats); otherwise the current configuration is kept. Viterbi
  ties within `maths.chains.CHAIN_TIE` take the fewest switches, then the current state, then the lowest; a genotype
  within `family_phase.map_solver.GENOTYPE_TIE = 1e-12` of the best keeps the current one. A sweep runs the blocks
  over a greedy colouring of the conflict graph (parent-child edges plus a clique of each root's children; samples by
  decreasing degree, then
  sample index): blocks of one colour share no free variable, so they run in parallel with the result independent
  of thread count and order. Then every root's relabelling moves (root, then position), the compound moves on pairs
  of at most `family_phase.map_solver.LONG_SPAN = 8192` sites, the compound relabellings and the site moves run
  sequentially in canonical order (sample order, then position). The long compound moves (longer pairs and terminal
  segments, kinds 0 and 1) form an optional second kind of sweep, `FamilyPhaseConfig.map_long_moves = False` by
  default: when enabled, one such sweep follows a sweep of the first kind that accepts nothing, and the solver returns
  to the first kind when it accepts anything. They are off because they cost 4-6 times 2.1's family-phase time on
  dense data and changed about 10 of 63 validated reference releases.
- **Four endpoints** (release 2.3). Internal sample coordinates use pedigree depth, roots first,
  followed by discrete scaffold and observed-read-mask keys. Pedigree-neighbour refinement separates
  otherwise identical keys. Sample names and likelihood floating-point values do not order the search.
  Both directions of this order run both schedules from the same default start. Every initial endpoint
  receives one continuation, for a total bound of four, before global ranking by exact J with the shared
  `maths.ties.tie_classes` definition and existing objective quantum. A worse initial endpoint can improve
  beyond the initially best endpoint during continuation; initial scores are diagnostic only.
  Each endpoint supplies its own coordinates and schedule throughout continuation, root-frame repair,
  polish and family evidence tests. Its transplant donor is the other schedule on the same axis.
  Differences are grouped by affected objective factors, including shared root marginals and selector
  transitions between supported markers across gaps. Independent gains are evaluated from one base;
  strict improvements are applied together. Diagnostics record tied configurations exceeding the
  continuation bound; unvisited configurations do not themselves mask released calls.
  Final score-tied alternatives are released in their native coordinates and aligned to the caller's
  axis before projection. Their disagreement is recorded rather than interpreted as evidence for a
  stored representative. The reversal switch changes execution order while preserving the endpoint
  set; it is not disabled. Residual identical keys from samples with no incoming or outgoing pedigree
  edges have independent unary and frame-chain factors; their storage order cannot couple their
  updates or releases, even when likelihood rows differ. Other duplicate groups require an
  exchangeability certificate (identical model rows and actual pedigree neighbours) and identical
  member-specific releases. If that certificate fails, this bounded solver reports that joint
  duplicate treatment is required; it does not order samples by names or likelihood floats.
- **Convergence and guarantee.** After every sweep the from-scratch J must have risen by the accepted gains
  (checked). Every accepted step raises J by more than the quantum, J is bounded and the configurations are finite,
  so each ascent stops when a sweep of each kind accepts nothing. Cycles are impossible. The release is a fixed point
  of the enabled block, relabelling, compound and site moves and within the shared objective tie tolerance of the best of all four
  continued endpoints; it is not a certified global maximum. `FamilyPhaseConfig.map_max_sweeps` is a safety cap per ascent;
  reaching it is an error.
- **Mendelian-conflict diagnostic.** `diagnostics.map_conflicts` lists the (parent, bp) sites where the converged
  configuration needs a transmission error (a root's best genotype, or a non-root's MAP genotype, leaves a child's
  allele mistransmitted).

From the converged configuration every sample's family MAP frame is released over its painting-bin runs: a root's
frame chain itself; any other sample's frame path decoded from the per-site Gibbs conditionals of its ordered
genotype (its potential, slot terms and own children's factors at the converged configuration, normalised) used as
`phase_view` emissions (frame switches at painting-bin edges with probability `phase_switch_probability`, ties by
`CHAIN_TIE`, or `FRAME_TIE` for a chain without an anchor). A point-mass genotype would resolve a tied order to
0|1 and act as false phase evidence. The family MAP frame is what recombination decodes for samples with fewer
than two observed parents; for samples with two observed parents the polish below gives the released and the
decoded frame. The solver keeps no work state: an interrupted chromosome is recomputed.

## Root-frame repair

A root's scaffold frame can be wrong over the rest of a component: panel rows exchanging founder identity, painting
abstentions or ties between interchangeable rows, segregating deletions. Every child then changes homologue near the
error. Where the root's own data cannot place that change, the children's changes spread over a stretch; the compound
relabelling covers changes within its window, beyond it the family MAP keeps the scaffold frame, and the MAP can
also change a root's frame where the scaffold is right. After the solve, `family_phase/frames.py` repairs the frames
of roots
with at least `FamilyPhaseConfig.root_frame_minimum_children = 3` children on the converged family MAP, without
changing it (a recomputed stage makes the same decisions).

- **Votes.** For root P, a bin edge e inside a component and each child edge of P, the child's joint selector
  chain is re-decoded with free ends after
  relabelling P's homologues from e on; its emissions at the family MAP use conditional cavities (every observed
  parent's genotype summed without the voting child's factor, the child's own genotype summed), so a vote is not
  confirmed by a parental genotype fitted to the voter itself. The exact vote is the change of the max-path log score
  on the window
  ±W around e (W = `FamilyPhaseConfig.root_frame_window_bp = 500000.0` bp); at a flagged edge it is the best such
  change at any boundary of the flagged stretch around e. The net vote is the same change with the evidence inside
  a location zone left out (e ± `FamilyPhaseConfig.root_frame_location_bp = 250000.0` bp, or the flagged stretch
  widened by that tolerance) and the zone crossed by one transition priced like e, so only the flanks count. A
  child supports a change near e if its exact vote reaches τ
  (`FamilyPhaseConfig.root_frame_informative_nats = 2.0`) and its net vote exceeds −τ; it opposes if its net
  vote is at most −τ and its exact vote agrees (the best relabelling inside the stretch costs at least τ; release
  2.2: a child whose exact vote supports the change while its flanks show no net change, a double switch inside the
  zone, is neither a supporter nor an opponent). Edges whose zone reaches a component end get no votes.
- **Flagged edges** are the edges where the root's own data do not carry the scaffold orientation: edges bounding
  or inside bins without any scaffold-heterozygous site of the root, and painting tie boundaries. Their prior
  log-odds of a scaffold frame change is 0; elsewhere it is log(p / (1 − p)) with p the phase switch probability.
- **Moves.** Candidates are the bin edges next to children's current selector changes, with at least three children
  changing inside the window. Where P's family MAP frame agrees with its scaffold over the whole net window, both
  are relabelled from e to the end of the component when at least three children support it and none opposes
  (a frame error moves every child, all or none, while real crossovers and reference-order junctions switch
  children binomially), the prior plus every child's exact gain is positive, the junction guard passes and e lies
  outside the protected intervals. The junction guard blocks e when another parent with at least three children of
  its own (children shared with P left out) is mixed inside the two painting bins next to e: among at least three
  decided children, at least two change and at least one does not. Proposals are taken in order of their total,
  one per root and net window in a round, for at most `FamilyPhaseConfig.root_frame_max_rounds = 8` rounds; each
  round sees the chains relabelled by the previous one.
- **Reverts and transfers.** At a change of P's family MAP frame against its scaffold (the only one in the net
  window) the same votes, guard and protection decide: at least three supporters, no opponent and a positive total
  revert the internal (decoder) frame, since the scaffold was right; at least three opponents, no supporter and a
  negative total copy the change into the scaffold (transfer), so the displayed frame agrees with the decoded one.
- **Children's frames.** A child with one observed parent releases its family MAP frame, which the family MAP
  aligned with the parent's homologue labels. When a root's internal frame changes, each such child's frame is
  changed with it inside every run of the parent's heterozygous sites at which the child is heterozygous (bounded
  by sites where the child is homozygous, which are data) that contains the change, so that its meiosis
  observations keep the family solve's pattern relative to the homozygous sites on both sides. Where the change
  reverses the parity between those sides, the run's own observation switch nearest to it is dropped, or, when the
  run has none, the switch at the change is kept, as the data require one (`follow`, `cancel` and `keep` records).
  Children with children of their own are recorded (`not_adjusted_parent`), not changed.

Scaffold changes (moves and transfers) enter before the polish, so the displayed phase, the polished children and
the recombination decoder share one root frame there. The product lists every decision (`FrameRepair`: moves,
reverts, transfers, child frames, rejection counts by reason, junction-guard blocks, flagged edges per root).

**Protected intervals.** `[family_phase] protected_intervals` names a breakpoint TSV in the reference-order
breakpoint format (header `chrom start end reason`, `#` comments, 1-based inclusive bp; see
[recombination](recombination.md)). No move, revert or transfer is made at a bin edge whose marker interval
overlaps a protected interval; nothing is masked. Rows whose reason starts with `reversed_segment` are not
protected intervals. The rows' content enters the stage identity.
<!-- owner question (BLUEPRINT 8, T8 Q3): protected intervals ignore reversed-segment rows while recombination treats both ends of a reversed segment as breakpoints -->

## Phase polish

The polish (`family_phase/polish.py`) starts from the repaired scaffold with every frame 0 and maximises a
conditional copy-consistency score over sample frames f[n, j] (which scaffold track is first) and per-edge
transmission selectors s[e, j]:

    score = Σ_e Σ_j copy(parent allele s ⊕ f_parent, child track slot ⊕ f_child)
          + Σ_e Σ_j selector switch terms + Σ_n Σ_j frame switch terms,

where copy is log(2(1 − c)) for a match, log(2c) for a mismatch and 0 when either allele is unknown, with
c = `FamilyPhaseConfig.polish_copy_error = 0.01`; only called scaffold alleles count, never genotype likelihoods or
family fills. Selector switches use the Haldane probabilities, 1/2 at component edges (the polish and the
recombination decoder restart selectors at assembly breaks). Initial selectors are each edge's MAP path given the
initial frames (back-pointers within `CHAIN_TIE` stay, final state 0 within `CHAIN_TIE`).

Frame switch probabilities refine the painting bins: every painting-bin edge inside a component is allowed, and
so is every marker interval within `FamilyPhaseConfig.polish_window_sites = 100` sites of a change of the initial
selector paths. Each painting-bin edge forms a group with the intervals of the bin that follows it (up to the next
bin edge or component edge); for n allowed intervals in a group θ = (1 − (1 − 2p)^(1/n))/2, so their composed
transition keeps the edge's probability p. Component edges switch with probability 1/2 and every other interval
has θ = 0: the first painting bin of a chromosome or component has no edge on its left, so the polish never
switches phase inside it.

Coordinate ascent runs over a greedy colouring of the pedigree graph (no two vertices of a colour adjacent; samples
by descending degree, then sample index; colours forward on even sweeps, backward on odd ones). Within a colour,
every child with two observed parents is re-decoded exactly by an eight-state Viterbi over (frame, two incoming
selectors) with its neighbours fixed, factorised over the three binary axes; its vertices share no variable, so
their parallel updates do not depend on the thread count. Tie rules (release 2.1): back-pointers within
`maths.chains.CHAIN_TIE` stay (fewest switches); the final state is the current one when it is within `CHAIN_TIE` of
the best, else the lowest such state. Each maximal interval where the new path differs from the current
configuration is accepted on its own exact gain (one routine, never the dynamic-programming maximum), only above the
quantum `FamilyPhaseConfig.polish_minimum_gain = 1e-06` nats (inside it the current configuration is kept) and,
where it changes an observable phase (a frame change at a scaffold-heterozygous site), only if its copy-consistency
gain is at least `FamilyPhaseConfig.polish_minimum_biological_gain = 0.0` minus that quantum (an exactly zero gain
passes whatever its rounding). The polish score depends only on called scaffold alleles and fixed switch
probabilities, so genotype-likelihood noise cannot move it. The polish must converge (a sweep accepting nothing)
within `FamilyPhaseConfig.polish_max_sweeps = 6` sweeps; otherwise the stage stops with an error. Samples with fewer
than two observed parents keep frame 0. Called alleles never change, only their track order.

## Frames and released calls

The released calls `calls int8[N, L, 2]` are the scaffold's two alleles at every site in the displayed track
order, so genotypes and missingness are exactly the scaffold's: no allele is imputed, filled from relatives or
re-called, and no separate imputation product exists. Two frames, relative to the painting's scaffold orientation
(1 = the scaffold's tracks swapped), are released per sample and site:

- `displayed_frame`: the order of `calls`. For a child with two observed parents it is the polished frame, so
  displayed track k is the copy transmitted by the parent in slot k of the released edges (the pedigree's Parent1 and
  Parent2 columns). For a child with one observed parent (release 2.2; up to 2.1 the scaffold orientation) it is
  oriented by parental origin: its track `slot` (slot 0 for a single observed parent) is the copy transmitted by the
  observed parent. Inside each stretch between the child's painting tie boundaries (and component edges) the
  painting's phase is kept, and the stretch is oriented by its family-MAP phase view: frame 1 when the stretch's summed
  frame log-likelihood ratio (the per-site `phase_view` evidence of the decoder frame) is positive, else the
  painting scaffold's orientation. So the phase across a painting tie boundary, where the painting's track order is a
  convention, is decided by the family data, while the painting's data-driven phase inside a stretch is not replaced
  by the decoder frame's bin-level switches (stress_n80 chr10: one-parent children's component-aligned allele errors
  272,122 -> 110,650 with 4 -> 4 phase switch errors; the decoder frame itself gave 164,610 and 27). For roots it is the
  repaired scaffold orientation (the moves and transfers of their root-frame repair), an orientation label without
  parental meaning.
- `decoder_frame`: the family model's internal frame, which recombination decodes (`decoder_calls()`). It equals the
  displayed frame for two-parent children; for roots and one-parent children it is the family MAP frame with the
  repair's moves, reverts and children's following frames applied.

**Released ambiguity** (release 2.2, `family_phase.stage.released_ambiguity`). Where the data cannot place a frame,
the product says so instead of releasing a convention:

- `frame_undetermined int8[N, L]`: bit 0 (value 1) = the decoder frame is undetermined at the site, bit 1 (value 2) =
  the displayed frame is. A frame is undetermined where the two extreme tie placements of the sample's phase chain
  differ, everything else held at the converged configuration: the default decode puts an unavoidable switch at the
  leftmost tied position and the mirrored decode (`maths.chains`, `reverse=True`) at the rightmost; a chain whose
  global frame is a tie is undetermined everywhere. This covers roots' MAP frame chains and one-parent children's
  phase views (decoder bit; a one-parent child's displayed stretches keep the painting's phase, so its displayed
  ambiguity is released as phase breaks), two-parent children's polished chains (`polish.undetermined_frames`, both
  bits), and a root-frame repair decision at a
  flagged edge, which could sit at any boundary of its flagged stretch (`FrameDecision.interval_left_bp` /
  `interval_right_bp` give that interval). A chain-placement ambiguity can lie entirely between informative calls (the release 2.2 TroMau chr7
  example had no heterozygous calls in its ambiguous stretches). Retained, evaluated family modes are also
  projected onto their released ordered calls: only differing heterozygous or half-called cells acquire the
  corresponding displayed or decoder uncertainty bit. Homozygous and fully missing cells do not differ under
  a strand swap. An unvisited continuation is a search limitation, not evidence for masking cells.
  Export writes heterozygous and half-called
  sites with the displayed bit set unphased; recombination censors non-homozygous calls with the decoder bit set.
  Validation therefore reports both ordinary callability and the fraction of called alleles whose displayed
  phase is supported, together with phase-set counts and physical-span N50.
- `phase_breaks` (+ `phase_break_offsets`): per sample, the marker boundaries where the displayed phase of the right
  side relative to the left side is not determined by data: a painting tie boundary (inside a component; the
  painting's track order there is a convention) of a root that the root-frame repair does not decide (no decision's
  stretch holds it and fewer than `root_frame_minimum_children` children oppose a change there with none
  supporting; every tie boundary of a root with fewer children), and a painting tie boundary of a one-parent child
  when the stretch on either side of it carries no frame evidence (its orientation is then the painting's).
  Two-parent painting boundaries also require a family evidence witness. The adjacent pieces are compared in
  all four absolute orientations, using the same affected family, start, genotype constraints and bounded refit.
  The best scores for equal and opposite relative orientations are compared with the shared tie definition.
  Both flanks must have supported orientation; a tied relative order, an unsupported flank, or conflicting
  retained alternatives produces a break. A joined boundary records its relative margin and both flank margins
  in `diagnostics.map_two_schedules["release"][...]["painting_joins"]`.
  For two-parent samples, long disagreements between the existing MAP phase view and polish also delimit
  phase sets. Each component-local disagreement run longer than the existing `LONG_SPAN` (8,192 markers)
  and containing a fully called heterozygote contributes its two internal endpoints. Whole-component strand
  swaps add no internal break. This comparison uses observable ordered calls: different frame-switch
  positions inside homozygous or fully missing stretches do not turn a whole-component swap into a relative
  phase disagreement. The disagreement set is unchanged when the two outputs are exchanged.
  This records disagreement between two estimators; it does not correct the point estimate or alter calls,
  frames, decoder uncertainty, or the objective J.
  Export starts a new phase set after each break (component edges already start one). Display-only breaks do
  not by themselves set decoder uncertainty or reset recombination selector chains.
  Export retains PS continuity across displayed-undetermined sites. After the final alternative projection,
  each internal undetermined stretch therefore requires a witness connecting its determined flanks in every
  retained branch. Two-parent fits use the same four-orientation comparison and leave undetermined target
  genotypes unconstrained. One-parent flanks reuse their signed per-site orientation evidence, accumulated
  in float64 with the existing chain tolerance. Roots reuse accepted repair votes covering the stretch.
  Unsupported or disagreeing witnesses add a break after the stretch. Whole-component or terminal uncertainty
  has no two-flank continuity claim. Diagnostics record the witnesses and the number of added breaks.

The fallback two-parent point estimate uses the family MAP through root-frame repair. One-parent displayed
stretches use the family-MAP orientation evidence described above. MAP also supplies the decoder frame of
samples with fewer than two observed parents and the diagnostic
`changed_observable_phase_sites` (two-parent sample-sites at scaffold-heterozygous sites where the polished frame
differs from the family MAP frame).

## Product and records

`FamilyPhaseProduct` (`family_phase/records.py`, record `family_phase` per contig):

| field | type | meaning |
| --- | --- | --- |
| `sample_ids` | N | run sample order |
| `positions` | `int64[L]` | every evidence site of the contig |
| `component_ids` | `int32[L]` | assembly component, −1 between components |
| `bin_edges`, `bin_edge_offsets` | `int64[M]`, `int64[C + 1]` | painting-bin edges (bp, half-open bins) per component; recombination reads its bins here |
| `calls` | `int8[N, L, 2]` | released alleles (−1 unknown, 0 REF, 1 ALT) in displayed order |
| `displayed_frame`, `decoder_frame` | `int8[N, L]` | the two frames above |
| `coverage`, `coverage_offsets` | `int64[R, 2]`, `int64[N + 1]` | each sample's painted site ranges [start, stop) |
| `edges_identity` | str | identity of the `ParentEdges` the phase is conditioned on |
| `diagnostics` | `FamilyPhaseDiagnostics` | family-MAP sweeps, converged, accepted steps per sweep, J after the initialisation and every sweep, polish updates per sweep, called alleles, changed observable phase sites, frame repair decisions, elapsed seconds, Mendelian-conflict sites (sample, bp) |
| `frame_undetermined` | `int8[N, L]` | released ambiguity of the decoder (bit 0) and displayed (bit 1) frames (release 2.2) |
| `phase_breaks`, `phase_break_offsets` | `int64[B]`, `int64[N + 1]` | per sample, marker boundaries where the displayed phase is not determined across (release 2.2) |

Evaluation and export read `sample_ids`, `positions`, `component_ids`, `calls` and the released ambiguity fields.
Export starts phase sets at component edges and explicit phase breaks; displayed-undetermined calls are unphased
without implicitly splitting the phase set. Evaluation scores those same sets. `OUTPUT/family_phase/summary.csv` has one row per contig: `samples`,
`sites`, `map_sweeps`, `converged`, `map_objective` (final J), `called_alleles`, `changed_observable_phase_sites`,
`root_frame_moves`, `root_frame_reverts`, `root_frame_transfers` and `elapsed_seconds`.

## Implementation and cost

- **Structure.** One chromosome per worker of a chromosome executor sharing the stage's CPU budget; the number of
  concurrent chromosomes is limited by a per-chromosome byte estimate (ten evidence-sized arrays, two decoded
  paintings and 5 GiB). Inside a chromosome, Numba kernels parallelise over the blocks of a colour, samples and
  child groups (objective, release) and the vertices of a polish colour. The four initial endpoints execute as
  independent tasks under the chromosome's live CPU and memory budget; after they join, the four continuations
  execute the same way. Each task owns its mutable state and shares immutable coordinate inputs. Thread claims
  are refreshed at kernel boundaries; completion order never determines ranking or the stored representative.
  Initialization parallelises only dependency waves preserving parent and shared-parent sibling precedence.
  Independent site columns run concurrently while parent order and gain/diagnostic folds stay fixed.
  Orientation witnesses use execution-layer tasks with marker-window scratch including their true flanking
  selector support and root transitions. Their scratch and masks use a compact family/factor graph, including complete shared-root factors and
  pinned exterior genotypes with their original non-root status. Unrelated sample rows are omitted. The private
  stage6s prototype supports `HAPLOTYPES_COMPOUND_SCHEDULE=serial` as the reference and `coloured` for a fixed
  sweep catalogue of factor-independent compound moves. Snapshot pairs are revalidated; new pairs wait for the
  next sweep. Relabel tails conflict over their full affected rows. This changes search scheduling, with J and
  every tolerance unchanged; promotion requires the accuracy and stability gates.
- **Cost.** The family model is independent of the founder count K. With bounded family degree, a fixed sweep
  has O(N L) sample-block work. Repeated evaluation of shared-root sibling factors can introduce a degree factor;
  graph setup and witness metadata also contain repeated sample/edge scans. The implementation does not yet
  establish near-linear scaling when family degree grows with N. Additional work includes the relabelling scan (per root, one binary search per child and frame-chain boundary)
  and the sequential compound moves, whose windowed re-decodes and gains touch only the moved family's blocks and
  terms; site moves enumerate a sibship's genotypes at its Mendelian-conflict sites only. The solver converges in at
  most a few sweeps (measured: at most 6). Root-frame repair re-decodes windows of the children's selector chains per
  candidate edge; the polish costs O(L) per two-parent child and sweep.
- **Memory.** Each workspace holds float64 potentials (N × L × 4), discrete scaffold/orientation
  arrays and an observed-read mask. The symmetric search retains two coordinate views in addition to the
  caller workspace when neither view is already in caller order, four initial discrete states, and up to
  four continued states. States contain int8 genotypes and frames (N × L) and selector states (G × L).
  Coordinate views share marker arrays but copy sample-indexed arrays. Initialisation temporarily holds
  float32 conditionals (L × 4) for each non-root parent; release uses repair/polish work arrays.
  Raw genotype likelihoods are released once the potentials exist. There are no belief-propagation
  message arrays. Peak RSS must include all views, retained endpoints and concurrent workers.
- **Resume.** `identity/family_phase` is bound per contig before any work and checked on resume; a contig with a
  `family_phase` record is not recomputed; an interrupted contig is recomputed.
- **Determinism.** Blocks of a colour share no free variable and colours run in a fixed order; relabelling and
  compound candidates use the endpoint's pedigree/data coordinates; each prototype colour is factor-independent
  and gains fold in catalogue order. Sums run in fixed orders, so results do not depend on the
  thread count.
- **Convention reversal** (`HAPLOTYPES_TIE_REVERSE=1`, release 2.3). Family phase always runs both
  explicit schedules in both pedigree/data coordinate directions. The global switch changes their
  execution order, while continuation, repair, polish and release evidence use the owning endpoint's
  explicit schedule. Within a mirrored schedule, chain directions, tied state labels, sample/colour/root
  scans, site enumeration, polish scans and repair tie order are reversed. Every retained release is
  aligned back to caller coordinates before projection. Fewest switches, keep-current inside a tie
  class, conservative threshold sides, descending degree, and scaffold-based starts remain stated
  priors. Recombination applies its own reversal checks after consuming this symmetric family product.

## Interpretation and limitations

- The released phase is a point estimate, not a calibrated marginal posterior; no probability tensors are
  produced.
- Everything is conditional on the released Tier-B edges: unreleased or missing parents are not imputed, and a
  one-parent child's other track is constrained only by its own scaffold.
- The scaffold limits what can be phased: a track the painting withholds, or whose candidate rows disagree, stays
  unknown even where relatives would determine it.
- The released family MAP is a fixed point of the enabled moves and within the shared objective tie tolerance of the best of all four continued endpoints,
  not a certified global maximum, and it can stop short of the best mode. Earlier two-schedule validation (19 Oct 2026, fixed upstream,
  measured with the long compound moves enabled): the same ascent started from a belief-propagation configuration
  (25 iterations, the method 2.1 replaced) ends higher than the release on 7 of 22 TroMau chromosomes (by at most 23.9
  nats; 2.1: 11 of 22, up to 303), on 2 of 8 `std_n80` chromosomes (at most 30.6; 2.1: 5 of 8, up to 153) and on 7 of
  8 `stress_n80` chromosomes (155 to 2,598 nats; 2.1: 8 of 8, 797 to 11,147), and lower on the rest (stress chr1 by
  1,162); with the default (long moves off) the gaps can be larger on the chromosomes where those moves were
  accepted. The remaining gaps are joint changes of an F1 root's frame with its descendants over long regions in the
  multi-generation stress pedigree, and a few multi-family genotype changes on TroMau.
- The symmetric four-endpoint search still explores a bounded subset of modes. Its diagnostics record
  alternatives that could not be jointly continued within the bound. Masks project only evaluated alternatives;
  these masks are not a calibrated posterior or a claim to enumerate every possible mode.
  The release 2.2 stage-level reversal comparison had differences on 8 of 38
  TroMau/std/stress chromosomes. Release 2.3 replaces the convention-dependent base and continuation policy;
  validation must compare released calls, uncertainty, phase sets and recombination together.
- Root-frame repair is a vote with fixed thresholds. A real scaffold frame error of a root with a single child
  becomes one false crossover call; the two cannot be told apart.
- Inside a painting bin the frame is constant except where the polish uses the fine boundaries near selector
  changes; within a bin the position of a phase change is otherwise not identified.


The stage6t performance path retains the stage6s model, schedules and tolerances.
Touched-factor lists replace chromosome-length masks in compound, site and
transplant scores; merging the sorted lists preserves the original floating-point
addition order. Difference clusters use an interval index for each factor and
return the same connected components. Pedigree depths use a DAG traversal;
root-child adjacency and sample partitions are indexed once. Each witness's four
orientations run as separate budgeted tasks from the identical common start,
sharing immutable local input arrays and owning their mutable states. The serial
compound reference remains available through HAPLOTYPES_COMPOUND_SCHEDULE=serial;
its value is part of the family checkpoint identity. No score threshold changes.

These changes remove the per-move global-mask factor and the all-pairs difference
cluster scan. Complete shared-root contributions still depend on family degree;
the near-linear claim is conditional on bounded family size. Coordinate refinement
can still take N rounds in a pathological unresolved pedigree. These remaining
terms are recorded in the performance design, rather than declared solved.

### Witness scratch admission and implicit root conflicts

Independent orientation configurations share the immutable family/window input
while each owns its mutable state. A flexible reservation from the execution
memory governor bounds all live templates and mutable scratch in one batch.
The requested allowance is one additional chromosome-sized mutable state or
one largest complete family window with all its configurations, whichever is
larger; the minimum is one configuration. Work waits for scratch and every
configuration still receives the same refits. Idle templates release their
arrays before returning memory credit. This affects scheduling, not evidence,
phase decisions or the number of evaluated alternatives.

Root sibling conflicts are stored as factor membership rather than explicit
clique edges. Shared co-parent multiplicities preserve the exact original
greedy degrees and visiting colours; direct edges and occupied factor colours
use linear storage in pedigree size. Repeated root-factor scoring and worst-case
canonical refinement remain separate scaling costs.


### Parallel execution and parental count cavities

Independent endpoint, retained release and witness tasks use the production CPU budget. Long child-block term tables parallelize across markers; root emission tables parallelize across chain runs while retaining the original summation order within each run. Colour membership, visit order and accepted-move rule are unchanged. Sparse colours can distribute spare threads inside their blocks. Whole-task memory reservations cover their nested work instead of reserving those arrays twice; the governor still uses available node memory and normal headroom. Witness batches retain their own scratch allowance.

Parental copy factors admit integer sufficient counts indexed by selected parental homologue and child allele. A cavity subtracts the excluded group's one or two slots from these counts. Initial counts contain only decoded child blocks, including the original soft-parent initialization. Mutable counters are private to each endpoint and update after a parallel colour or initialization wave; concurrent siblings never write one counter simultaneously. Constrained witness refits refresh after joint genotype updates and update after each child block. Static conditional and repair queries use a cache of the converged state. The mathematical objective J and every tolerance remain unchanged. Count-times-log arithmetic changes rounding from the sequential fold, so scientific decisions require the release gates. HAPLOTYPES_CAVITY_MODE=fold retains the reference arithmetic; the mode participates in checkpoint identity.

Canonical pedigree refinement uses deterministic smaller-piece splitters with keys from the existing discrete scaffold, pedigree and slot data. It retains the equivalence partition and does not use sample names. Refinement costs O((N+E) log N); split-class ordering can change search endpoints and is validated as such.

Prototype effort bounds are disabled by default pending their standard-data continuity review. HAPLOTYPES_FAMILY_WORK_BOUND bounds a complete witness neighbourhood, including shared root marginals and pinned boundary factors. Overflow releases a display phase-set break with computational_bound provenance and empty scores. It does not truncate a family or change calls, decoder frames or ambiguity masks. The separate HAPLOTYPES_COMPOUND_WORK_BOUND records uncomputed optional proposals and adds display breaks at their proposed tract boundaries. Omitting search proposals can change the MAP endpoint and decoder and therefore requires the stronger decision gate. Zero selects the unbounded reference for either bound; both values participate in checkpoint identity. Display-only breaks do not reset recombination selector chains.


### Independent sites in the witness refit

For a fixed parent, the joint genotype update conditions on selectors and frame states. It updates one marker at a time and reads no genotype at another marker. The implementation evaluates disjoint marker chunks concurrently, with private enumeration scratch, then joins them before advancing to the next parent. Parent visitation, configuration enumeration, the finite-state limit, tie rules and the number of refit sweeps are unchanged. A configuration claims its live CPU allowance at this kernel boundary.

Copy-factor sufficient-count queries use four scalar integer counts. Child-term and conditional loops reuse a private four-state root scratch buffer across sites instead of allocating it for each query. These are storage changes; the arithmetic expressions and accumulation order are unchanged.

The complete-family witness and optional-compound effort bounds remain disabled. Release 2.3 retains the complete evaluated family context; a 64-sample default was rejected because it would remove all two-parent witness decisions on the standard N=320 simulations. Scaling of the unbounded witness remains a measured acceptance requirement.

Root marginal site factors use four scalar accumulators, each retaining the original child-incidence addition order, rather than a newly allocated buffer for each evaluation. Budgeted sparse-colour and witness blocks may evaluate independent marker emissions and genotype choices concurrently; the selector Viterbi and carry-forward order are unchanged. Outer parallel block callers retain the serial marker path. All paths use the same genotype tie rule and exact within-site operations.

The four-state log-sum-exp kernel omits exponential calls for impossible genotypes (`-inf` potential), whose terms are exactly zero when the maximum is finite. It retains the original left-associated addition order. A single finite state gives the maximum plus `log(1) = 0`; the implementation preserves that addition, including signed zero. No finite likelihood term is removed or approximated.


### Exact first-pass sharing and chromosome scheduling (2.3)

For four configurations of two disjoint flanking pieces on one sample, the
first joint-genotype pass has only two possible complete input columns at each
marker. The scorer evaluates the complete parent-ordered pass for the unflipped
and both-flipped configurations, then assembles the matching columns for the
other two. Shared root factors and non-root parent dependencies remain in each
pass. Initial objective scores, subsequent selector/root refits and convergence
checks are unchanged. Overlapping pieces, different samples and other sets of
configurations retain the independent reference calculation. Two additional
family-local genotype windows are charged to the scratch reservation. These are
computational reuse rules; the estimator, phase sets and recombination evidence
do not change.

Chromosome CPU-work weights are separate from memory reservations: marker count
plus the minimum two-sided compound window per painting switch. Longest-first
partitioning starts large work first, with initial leases proportional to each
worker's first task rather than its queued tasks. Initial process concurrency leaves room
for four endpoints with three numerical threads each, using the full allocated
CPU budget through live leases; freed cores move to remaining work at kernel
boundaries. This scheduling estimate is name-free and has no effect on candidate
sets, arithmetic or release rules. Node memory retains its automatic available
memory limit and normal headroom.

Scratch admission precedes numerical CPU leases: each configuration batch fits
its reservation even with all private states live. The two reusable first passes
and then the four continuations run through the same live thread budget. This
retains configuration parallelism when only one large witness fits in memory,
without letting tasks waiting for scratch occupy otherwise useful CPU shares.
