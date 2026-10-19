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
- **Two schedules** (release 2.2). The ascent's end point can depend on its schedule (block order, scan direction,
  tie conventions). The released MAP therefore runs the ascent (first kind of sweep only) from the default start
  under the canonical schedule and under its mirror (every arbitrary order and tie convention reversed, as the
  convention-reversal switch does),
  takes the end point with the higher J as the base and transplants into it, best exact gain first, each cluster of
  differing cells of the other end point (cells linked by family and site proximity) whose exact change of J exceeds
  the quantum, re-scoring the rest after each transplant (exact ties within the quantum: the leftmost cluster,
  counted); then the ascent runs again to convergence (with the long compound moves when `map_long_moves`). Both end points are optima of the same J, so this is a fixed
  pair of deterministic starts with exact recombination, not a different estimator.
- **Convergence and guarantee.** After every sweep the from-scratch J must have risen by the accepted gains
  (checked). Every accepted step raises J by more than the quantum, J is bounded and the configurations are finite,
  so each ascent stops when a sweep of each kind accepts nothing. Cycles are impossible. The release is a fixed point
  of every move above (block-, relabelling-, compound-, site-wise optimal) and at least as good as both schedules'
  end points; it is not a certified global maximum. `FamilyPhaseConfig.map_max_sweeps` is a safety cap per ascent;
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
  `interval_right_bp` give that interval). Undetermined stretches hold no heterozygous site whose phase the data
  determine (measured on TroMau chr7: 566,310 undetermined cells, 0 at heterozygous calls, 100 at half calls); export
  writes a heterozygous or half-called site with bit 1 set unphased.
- `phase_breaks` (+ `phase_break_offsets`): per sample, the marker boundaries where the displayed phase of the right
  side relative to the left side is not determined by data: a painting tie boundary (inside a component; the
  painting's track order there is a convention) of a root that the root-frame repair does not decide (no decision's
  stretch holds it and fewer than `root_frame_minimum_children` children oppose a change there with none
  supporting; every tie boundary of a root with fewer children), and a painting tie boundary of a one-parent child
  when the stretch on either side of it carries no frame evidence (its orientation is then the painting's).
  Export starts a new phase set after each break (component edges already start one).

The family MAP therefore enters the released calls only through the root-frame repair decisions; it gives the
decoder frame of samples with fewer than two observed parents and the diagnostic
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

Evaluation and export read `sample_ids`, `positions`, `component_ids` and `calls`; export writes them as phased
genotypes with one phase set per component. `OUTPUT/family_phase/summary.csv` has one row per contig: `samples`,
`sites`, `map_sweeps`, `converged`, `map_objective` (final J), `called_alleles`, `changed_observable_phase_sites`,
`root_frame_moves`, `root_frame_reverts`, `root_frame_transfers` and `elapsed_seconds`.

## Implementation and cost

- **Structure.** One chromosome per worker of a chromosome executor sharing the stage's CPU budget; the number of
  concurrent chromosomes is limited by a per-chromosome byte estimate (ten evidence-sized arrays, two decoded
  paintings and 5 GiB). Inside a chromosome, Numba kernels parallelise over the blocks of a colour, samples and
  child groups (objective, release) and the vertices of a polish colour.
- **Cost.** The family model is independent of the founder count K. A sweep costs O(N L) over the sample blocks,
  each site's maximisation touching the sample's own copy factor and its children's (O(E L) in all; a root's R sums
  over its children), plus the relabelling scan (per root, one binary search per child and frame-chain boundary)
  and the sequential compound moves, whose windowed re-decodes and gains touch only the moved family's blocks and
  terms; site moves enumerate a sibship's genotypes at its Mendelian-conflict sites only. The solver converges in at
  most a few sweeps (measured: at most 6). Root-frame repair re-decodes windows of the children's selector chains per
  candidate edge; the polish costs O(L) per two-parent child and sweep.
- **Memory.** Float64 potentials (N × L × 4), int8 genotypes, frames (N × L) and dense selector states (G × L),
  and during the initialisation float32 conditionals (L × 4) of each non-root parent; no message arrays. The
  genotype likelihoods are released once the potentials exist.
- **Resume.** `identity/family_phase` is bound per contig before any work and checked on resume; a contig with a
  `family_phase` record is not recomputed; an interrupted contig is recomputed.
- **Determinism.** Blocks of a colour share no free variable and colours run in a fixed order; relabelling and
  compound moves run sequentially in canonical order; sums run in fixed orders, so results do not depend on the
  thread count.
- **Convention reversal** (`HAPLOTYPES_TIE_REVERSE=1`, `maths.ties.reverse_conventions`; release 2.2 acceptance
  test). Every arbitrary convention of family phase runs reversed: chain decodes are mirrored (an unavoidable switch
  lands at the rightmost tied position) and state ties go to the highest state (selectors, genotypes, the polish's
  eight states, the end state of an unanchored phase view); sample, colour, root, boundary and pair scan orders,
  site-move enumeration, the polish's colour order and sweep direction, the repair's order among equal totals and
  its children's-frame ties, the shared-orientation candidate order and the junction table's ties are reversed. Not
  reversed (stated priors and inherited labels): fewest switches, keep-current inside a tie class, conservative
  threshold sides, descending degree, and start states that keep the scaffold orientation (the painting's track
  order, reversed by the painting's own switch).

## Interpretation and limitations

- The released phase is a point estimate, not a calibrated marginal posterior; no probability tensors are
  produced.
- Everything is conditional on the released Tier-B edges: unreleased or missing parents are not imputed, and a
  one-parent child's other track is constrained only by its own scaffold.
- The scaffold limits what can be phased: a track the painting withholds, or whose candidate rows disagree, stays
  unknown even where relatives would determine it.
- The released family MAP is a fixed point of every move above and at least as good as both schedules' end points,
  not a certified global maximum, and it can stop short of the best mode. Validation (19 Oct 2026, fixed upstream,
  measured with the long compound moves enabled): the same ascent started from a belief-propagation configuration
  (25 iterations, the method 2.1 replaced) ends higher than the release on 7 of 22 TroMau chromosomes (by at most 23.9
  nats; 2.1: 11 of 22, up to 303), on 2 of 8 `std_n80` chromosomes (at most 30.6; 2.1: 5 of 8, up to 153) and on 7 of
  8 `stress_n80` chromosomes (155 to 2,598 nats; 2.1: 8 of 8, 797 to 11,147), and lower on the rest (stress chr1 by
  1,162); with the default (long moves off) the gaps can be larger on the chromosomes where those moves were
  accepted. The remaining gaps are joint changes of an F1 root's frame with its descendants over long regions in the
  multi-generation stress pedigree, and a few multi-family genotype changes on TroMau.
- The two-schedule estimator is not exactly symmetric under the convention reversal (its base choice and final
  ascent follow the convention): in the stage-level reversal test 30 of 38 TroMau/std/stress chromosomes released
  identical outputs outside the released ambiguity, and 8 differed in up to 4 call cells and 4 crossovers (end points
  that neither schedule's basin reaches). It is a point estimate (overconfident relative to marginals).
- Root-frame repair is a vote with fixed thresholds. A real scaffold frame error of a root with a single child
  becomes one false crossover call; the two cannot be told apart.
- Inside a painting bin the frame is constant except where the polish uses the fine boundaries near selector
  changes; within a bin the position of a phase change is otherwise not identified.
