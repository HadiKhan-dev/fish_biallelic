# Family phase

Family phase (stage `family_phase`, entry point `family_phase.stage.run`) gives every sample a final,
pedigree-conditioned phase of every chromosome while preserving its called genotypes and missingness. It starts from
a phase scaffold decoded from the [painting](painting.md) and the [founder refit](founder_refit.md), runs damped
loopy belief propagation of a family model conditioned on the supported Tier-B parent edges released by [pedigree
inference](pedigree.md), repairs pedigree roots' scaffold frames by the votes of their children, and polishes the
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
| writes | `work/<contig>/family_phase` | transient belief-propagation state |
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

Hard-homozygote point masses keep fixed beliefs whatever the incoming messages, and factor work lists only the
(child, site) entries where some member is not a hard homozygote.

## Belief propagation and release

One iteration updates the phase chains, sweeps every copy factor (the order within a 32-site tile alternates with the
iteration's parity) and updates the selector chains. Messages are stored in float32 (copy-factor messages as
logarithms) and updated with damping in probability space, `FamilyPhaseConfig.damping = 0.5` being the weight given
to the updated message. The residual of an iteration is the largest message change over all three updates.

Every `FamilyPhaseConfig.gauge_move_interval = 5` iterations, joint root/children gauge moves
(`family_phase/gauges.py`) may relabel a root's homologues from one boundary on: this flips the root's frame and
every child's selector bit for that parent together. Per root, the MAP frame path and its children's MAP selector
paths are decoded, and a relabelling at a boundary is made when it raises their joint MAP score (frame-switch and
meiosis log-odds) and at least `FamilyPhaseConfig.minimum_gauge_support = 2` children change selector there; one
child cannot tell a root frame change from its own crossover. An iteration with a gauge move counts as unconverged.

The release state is the one after `FamilyPhaseConfig.max_iterations = 25` iterations, or after the first earlier
iteration whose residual is below `FamilyPhaseConfig.tolerance = 0.0001` without a gauge move. The release does not
wait for convergence: when the residual is still at or above the tolerance at iteration 25, the state of iteration
25 is used and the product records it as not converged, with every iteration's residual. From the release state,
every sample's family MAP frame is decoded over its painting-bin runs (`phase_view`): roots with their phase
messages, other samples with a neutral one. The family MAP frame is what recombination decodes for samples with
fewer than two observed parents; for samples with two observed parents the polish below gives the released and the
decoded frame.

The messages are saved as the chromosome's work state after an iteration whenever
`family_phase.records.WORK_SAVE_SECONDS = 120.0` seconds have passed since the last save, bound to the stage
identity; an interrupted chromosome resumes from them (large selector caches are rebuilt), and the work state is
cleared once the product is stored.

## Root-frame repair

A root's scaffold frame can be wrong over the rest of a component: panel rows exchanging founder identity, painting
abstentions or ties between interchangeable rows, segregating deletions. Every child then changes homologue near the
error. Where the root's own data cannot place that change, the children's changes spread over a stretch, so a
single-boundary gauge move gains nothing and belief propagation settles on the scaffold frame; it can also change a
root's family MAP frame where the scaffold is right. After the release, `family_phase/frames.py` repairs the frames
of roots with at least `FamilyPhaseConfig.root_frame_minimum_children = 3` children on the final family state,
without changing the messages (a resumed stage makes the same decisions).

- **Votes.** For root P, a bin edge e inside a component and each child edge of P, the child's joint selector
  chain (the family model's copy-factor emissions and selector transitions) is re-decoded with free ends after
  relabelling P's homologues from e on. The exact vote is the change of the max-path log score on the window
  ±W around e (W = `FamilyPhaseConfig.root_frame_window_bp = 500000.0` bp); at a flagged edge it is the best such
  change at any boundary of the flagged stretch around e. The net vote is the same change with the evidence inside
  a location zone left out (e ± `FamilyPhaseConfig.root_frame_location_bp = 250000.0` bp, or the flagged stretch
  widened by that tolerance) and the zone crossed by one transition priced like e, so only the flanks count. A
  child supports a change near e if its exact vote reaches τ
  (`FamilyPhaseConfig.root_frame_informative_nats = 2.0`) and its net vote exceeds −τ; it opposes if its net
  vote is at most −τ. Edges whose zone reaches a component end get no votes.
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
- **Children's frames.** A child with one observed parent releases its family MAP frame, which belief propagation
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
initial frames.

Frame switch probabilities refine the painting bins: every painting-bin edge inside a component is allowed, and
so is every marker interval within `FamilyPhaseConfig.polish_window_sites = 100` sites of a change of the initial
selector paths. Each painting-bin edge forms a group with the intervals of the bin that follows it (up to the next
bin edge or component edge); for n allowed intervals in a group θ = (1 − (1 − 2p)^(1/n))/2, so their composed
transition keeps the edge's probability p. Component edges switch with probability 1/2 and every other interval
has θ = 0: the first painting bin of a chromosome or component has no edge on its left, so the polish never
switches phase inside it.

Coordinate ascent runs over a greedy colouring of the pedigree graph (no two vertices of a colour adjacent; colours
in alternating order per sweep). Within a colour, every child with two observed parents is re-decoded exactly by
an eight-state Viterbi over (frame, two incoming selectors) with its neighbours fixed, factorised over the three
binary axes. When the best path raises the score by more than `FamilyPhaseConfig.polish_minimum_gain = 1e-06`
nats, each maximal interval where it differs from the current configuration is accepted if its own gain exceeds
that minimum and, where it changes an observable phase (a frame change at a scaffold-heterozygous site), its
copy-consistency gain is at least `FamilyPhaseConfig.polish_minimum_biological_gain = 0.0`. The polish must
converge (a sweep accepting nothing) within `FamilyPhaseConfig.polish_max_sweeps = 6` sweeps; otherwise the stage
stops with an error. Samples with fewer than two observed parents keep frame 0. Called alleles never change, only
their track order.

## Frames and released calls

The released calls `calls int8[N, L, 2]` are the scaffold's two alleles at every site in the displayed track
order, so genotypes and missingness are exactly the scaffold's: no allele is imputed, filled from relatives or
re-called, and no separate imputation product exists. Two frames, relative to the painting's scaffold orientation
(1 = the scaffold's tracks swapped), are released per sample and site:

- `displayed_frame`: the order of `calls`. For a child with two observed parents it is the polished frame, so
  displayed track k is the copy transmitted by the parent in slot k of the released edges (the pedigree's Parent1 and
  Parent2 columns). For samples with fewer than two observed parents (roots and one-parent children) it is the
  repaired scaffold orientation (the moves and transfers of their root-frame repair), an arbitrary orientation label.
- `decoder_frame`: the family model's internal frame, which recombination decodes (`decoder_calls()`). It equals the
  displayed frame for two-parent children; for roots and one-parent children it is the family MAP frame with the
  repair's moves, reverts and children's following frames applied.

The belief-propagation messages therefore enter the released calls only through the root-frame repair decisions;
they give the decoder frame of samples with fewer than two observed parents and the diagnostic
`changed_observable_phase_sites` (two-parent sample-sites at scaffold-heterozygous sites where the polished frame
differs from the family MAP frame).
<!-- owner question (BLUEPRINT 8, T8 Q1): the family BP is released unconverged at iteration 25 and the released calls do not depend on its messages beyond the frame repair; what should the BP contribute -->

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
| `diagnostics` | `FamilyPhaseDiagnostics` | iterations, converged, residuals, polish updates per sweep, called alleles, changed observable phase sites, frame repair decisions, elapsed seconds |

Evaluation and export read `sample_ids`, `positions`, `component_ids` and `calls`; export writes them as phased
genotypes with one phase set per component. `OUTPUT/family_phase/summary.csv` has one row per contig: `samples`,
`sites`, `iterations`, `converged`, `final_residual`, `called_alleles`, `changed_observable_phase_sites`,
`root_frame_moves`, `root_frame_reverts`, `root_frame_transfers` and `elapsed_seconds`.

## Implementation and cost

- **Structure.** One chromosome per worker of a chromosome executor sharing the stage's CPU budget; the number of
  concurrent chromosomes is limited by a per-chromosome byte estimate (nine evidence-sized arrays, two decoded
  paintings and 5 GiB). Inside a chromosome, Numba kernels parallelise over samples, 256-site belief tiles,
  32-site factor tiles, child groups and the vertices of a polish colour.
- **Cost.** The family model is independent of the founder count K: one iteration costs O(N L) for beliefs and
  phase chains, O(E L) over the supported factor entries and O(G L) for the four-state selector chains (G children
  with an observed parent). Selector chains are kept in 256-marker blocks; only blocks whose emissions changed are
  recomputed, propagating until the block-boundary messages are exactly unchanged, so every stored value equals a
  full recomputation. Root-frame repair re-decodes windows of the children's selector chains per candidate edge;
  the polish costs O(L) per two-parent child and sweep.
- **Memory.** Float32 messages (E × L × 4 to parents, G × L × 4 to children and selector cavities, N × L phase),
  float32 potentials and beliefs (N × L × 4), the float64 genotype likelihoods and the selector caches, which are
  bounded by `family_phase.selector_chains.MAX_SELECTOR_CACHE_BYTES = 25769803776` (24 GiB) per chromosome; groups
  beyond that budget are recomputed in full each iteration.
- **Resume.** `identity/family_phase` is bound per contig before any work and checked on resume; a contig with a
  `family_phase` record is not recomputed; an interrupted contig resumes belief propagation from its work state.
- **Determinism.** Updates run in fixed orders (copy factors per tile, chains per group, sequential score sums), so
  results do not depend on the thread count.

## Interpretation and limitations

- The released phase is a point estimate, not a calibrated marginal posterior; no probability tensors are
  produced.
- Everything is conditional on the released Tier-B edges: unreleased or missing parents are not imputed, and a
  one-parent child's other track is constrained only by its own scaffold.
- The scaffold limits what can be phased: a track the painting withholds, or whose candidate rows disagree, stays
  unknown even where relatives would determine it.
- The release point is a fixed iteration count. Samples with fewer than two observed parents are decoded by
  recombination in their family MAP frame, which therefore depends on the unconverged state when belief
  propagation has not converged.
- Root-frame repair is a vote with fixed thresholds. A real scaffold frame error of a root with a single child
  becomes one false crossover call; the two cannot be told apart.
- Inside a painting bin the frame is constant except where the polish uses the fine boundaries near selector
  changes; within a bin the position of a phase change is otherwise not identified.
