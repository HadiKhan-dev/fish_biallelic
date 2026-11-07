# Selection performance at low depth

Selection has two distinct searches. The feedback-round candidate bank uses
discovery's fixed-K fits and cavity scores, then a BIC neighbourhood search.
The path search fits starts and drop, merge and add proposals under the copying
model. Each fit can perform GEM updates, frequency-only updates and context
snaps. Their definitions and retained stopping rules are in
[selection](../methods/selection.md); complexity exemptions are in
[costs](costs.md).

## Evidence and scope

A matched census covers all 1,647 chromosome-1 blocks in each of the initial,
L1 and L2 rounds at depths 5, 1, 0.5 and 0.1, with 320 simulated samples.
It reads committed block summaries and traces, including completed blocks in
an unfinished low-depth run. These are chromosome-specific measurements, not
whole-genome estimates. Mean values are:

| Depth | Initial selected K | L1 bank fitted starts | L2 bank fitted starts | L2 path rounds |
|---|---:|---:|---:|---:|
| 5 | 5.57 | 6.96 | 6.95 | 1.00 |
| 1 | 5.07 | 23.89 | 23.93 | 1.01 |
| 0.5 | 4.68 | 37.55 | 39.07 | 1.02 |
| 0.1 | 2.87 | 69.14 | 101.07 | 1.31 |

At 0.1 the L2 bank fits up to 1,452 starts in an exceptional block;
selected K alone misses this temporary work. The census has no selected-fit
update backstops, path-round backstops or proposal-cut overflows. Bank budgets
bind in five L1 blocks and eleven L2 blocks at 0.1, and none at the other three
depths. These endpoint receipts do not describe the termination of every
rejected fit.

Instrumented saved-input replays separate iteration work from bank size.
For matched block 200, the initial round performs 75, 151, 215 and 776 GEM
updates at depths 5, 1, 0.5 and 0.1. Its discovery starts have 6, 6, 6 and 2
rows. At 0.5 the L1 replay spends about 6% of its time constructing the bank;
copying-model fits dominate. At 0.1 this block also has a bank tail: L1 fits
735 bank starts. Its L1 copying fits perform 864 GEM updates, 77 trajectory
frequency updates and 1,295 context-snap frequency updates. The bank and
copying fits both contribute materially. Context description/matching takes
less than 1% in this example. Thus neither larger selected K nor the context
matching exemption alone explains the increase.

Finer counters on initial blocks 600, 1000 and 1400 show 11 exact single-bit
escapes in total at depth 5, versus 1,704 at 0.1. Frequency-only continuations
over the same blocks rise from 178 to 260. Much of the extra ordinary-GEM work
is therefore allele escape steps, rather than frequency-only convergence.
Context snaps add their own one-bit changes and frequency refits.

The replay restores each round's saved incumbent, earlier calls, context
proposal and exact initial/L1 memo. Missing likelihoods are neutralised through
the same logical evidence projection as production. An uninstrumented call on
the same inputs is the reference for diagnostic instrumentation. Replacing the
saved memo with a cold fit would test a different execution contract
([LS13](../methods/selection.md#ls13)).

## Measuring the driver

Retain both feedback-bank and path-search receipts:

- Discovery latent K, alternative panels, completed proposal rows, bank rows,
  number and sizes of starts, and distinct searched starts.
- Per-search round K, proposal counts by kind, whole-class refit admissions,
  overflows and actual refit count. A retained incumbent can carry an older
  trace; its trace is not a count of work attempted in the current round.
- Updates and termination of every executed fixed-K fit, including rejected
  trials; frequency-only continuations, trajectory frequency updates, context
  snaps and their frequency updates.
- Actual E-step passes and cache hits by statistics level. Calls to the state
  accessor include hits and must not all be counted as new passes.
- Separate inclusive timers for candidate-bank construction, bank fixed-K fits,
  cavity scoring, path fitting, context description/matching and release.
  Nested inclusive timers must not be summed as disjoint work.

Wall clocks and detailed diagnostic lists belong in the lab replay, outside
scientific decisions and checkpoint identities. Existing bounded result
summaries remain useful for broad censuses. Compare logical work before
attributing timings on different or shared hosts to algorithmic changes.

## Optimisation boundary

The measured work motivates testing joint exact allele escapes, reducing their
repeated frequency refits, and avoiding duplicate model searches in candidate-bank
construction.
Both can change the optimisation trajectory and selected panel, even if the
objective is retained; they require a separately validated scientific change.
Reducing budgets, relaxing tolerances or pruning by a proposal's starting
likelihood has no identical-output justification: refits can change alleles,
and current screening includes lower bounds and complete tie classes.

A memo of context descriptions is mathematically safe when keyed by the exact
panel, context rows, called masks and matching convention. Its possible gain is
limited by the measured matching fraction. It does not resolve the copying-fit
or candidate-bank costs. Preserve the incumbent/memo completion precedence,
whole-class admissions, conditional one-bit checks and context rescue witnesses.

## Release 4.6 outcomes

Measured with the lab replay of 28 recorded chr1 blocks (three rounds each, one thread, warm kernels), N = 320:
- Selection wall time per block against 4.5: 0.79 (5x), 0.63 (1x), 0.56 (0.5x), 0.53 (0.25x), 0.40 (0.1x).
- Where the time goes now (warm profile): the E-step kernel is 81% at 1x and 83% at 0.1x. Candidate-bank kernels
  take 8-12%, release statistics 1-4%, and Python 3-4%. Python-to-Numba work therefore has at most a few percent left.
- Pass cost grows as K^2 (about 4.4 ns per pair state, sample and site for a counts pass). After 4.6 an exact pass
  costs about 1.3 flips passes (2.5 before).

Adopted, with their trajectory effects stated in the changelog:
- the joint GEM step (after a first frequency step: flips chosen at a start's guessed frequencies sent one block's
  K = 5 refit to a fixed point 265 nats worse);
- batched exact escapes and context snaps;
- SQUAREM fixed-panel refits;
- ranked bank starts;
- the exact-gain ratio chain.

Measured and rejected:
- **Read-free-site skipping:** exact, but 0.62x speed at 1x and 0.92x at 0.5x; it wins only below about 0.3x.
- **512-bit vectors:** bit-identical, 5-20% slower.
- **Exact escapes only at frequency convergence:** after the ratio chain it saves about 3.7 ms per tail iterate but
  costs a full pass per converging update.
- **Fusing the forward normalisation sweep into the next site:** the sweep is 11-14% of a forward pass at K = 4-6,
  and the audited per-lane reciprocal rule would change.
- **Joint steps without a first frequency step:** the 265-nat miss above, with one-founder and multi-founder flips.
- **Single-founder joint steps:** closest to 3.5 on the A35 fixtures, but worse on average on the replays.

Fastmath and SVML stay off, for cross-machine reproducibility.
