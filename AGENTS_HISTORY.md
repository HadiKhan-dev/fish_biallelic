# AGENTS_HISTORY.md

Why the rules in `AGENTS.md` exist: the owner's rulings in their own words and the incidents behind them, by date (all
2026). `AGENTS.md` cites an entry by its date tag, such as [15 Nov]. Read an entry when a rule's reason matters; agents
need not load this file every session. Rules live only in `AGENTS.md`; nothing here adds to them.

## September

- **16 Sep** (§7.4). `failed to register synthetic bubblewrap mount target /tmp/.git: No space left on device (os error
  28)` was cured by persisting the Codex runner's `TMPDIR` and restarting the runner. It is not the namespace error and
  does not mean the project filesystem is full.
- **27 Sep** (§6.6, §6.4). Observed: the controller rate limit was a 40-token bucket refilled by eight tokens a minute
  (a limit, not a target rate). Staggered, mostly 48-CPU two-hour `icelake-himem` workers reached 688 running SL4 CPUs
  across 14 jobs, with 96 more pending (an example, not an entitlement).
- **29 Sep** (§6.4). Observed: `icelake-himem` had `PriorityTier=2`, `icelake` and `sapphire` tier 1; within a tier an
  older array's age-based priority can put it ahead of a new request.

## October

- **4 Oct** (§6.5, §6.7). Whole-genome runs never use less than one full node. Keep a managed allocation through task
  boundaries when known follow-on work will use it: a two-hour worker that finishes its first task in 12 minutes keeps
  the remaining 1 h 48 min for the next tasks.
- **5 Oct** (§2.5, §4.1). One pipeline that works up to K = 30 founders. Match the process to the change: full process
  for model changes, one focused implementation and check for simple fixes.
- **7 Oct** (§6.4). "don't free up SL4 until they naturally expire, I don't get charged for those".
- **15 Oct** (§7.2). A cleanup loop over `ps -u "$USER" -o pid= --ppid PID` signalled every process the user owned on
  the hosting node (procps combines selection options as a union). It killed the `screen` session that kept the hosting
  job alive and so ended every agent.
- **17 Oct** (§3.1, §6.2). "real data aul stu is very very bloated, do not use it for any testing". "Don't request SL3
  nodes unless I tell you to ... request SL4 nodes only, they will come online eventually".
- **19 Oct** (§5.5, §6.4). "we aren't made of compute you know. Please don't go overboard on validations and tests and
  runs". Check `hpcstat` before choosing a partition.
- **20 Oct** (§2.3). "I want scientific result identicality, and if some bits shift around due to getting faster that's
  100% perfectly fine".
- **21 Oct** (§5.3, §4.5, §5.4, §5.5, §8). A delegated replay hung for two hours on a paid node: its
  `__main__`-defined workers could not be unpickled and the parent waited forever; the agent had ended its turn and the
  coordinator watched only exit codes. The owner asked that "the nodes we have allocated to us aren't being left idle or
  underutilised". A change that forces revalidation cancels the old validation in the same step. Delegated agents never
  spawn sub-agents.
- **22 Oct** (CLAUDE.md). Claude subagents run at effort xhigh, not max.
- **26 Oct** (§2.6, §8). Validate each fix where it acts; one end-to-end run per release candidate. Asking a reviewer
  never replaces your own work on the problem.
- **29 Oct** (§5.4, CLAUDE.md). The node queue `nodeq` serves every worker node; the idle-node hook was added.
  Prototype evaluations ran as one 8-core task per design, each looping over chromosomes and samples on about one core;
  each took about an hour while about 900 node cores sat idle. Split per chromosome, with a process pool over samples,
  the same work took minutes.
- **29-30 Oct** (§5.4). The second half of a TroMau validation was queued to wait about 12 hours for a worker node while
  the hosting node (a full 112-CPU node, 500 GB, 4.5 hours left) sat idle. On the owner's word it ran there instead.
- **30 Oct** (§2.4). "G0 can be allowed as parents here, but that's because they are present in the dataset we feed in
  rather than because their labelling is G0 and the F1s are labelled F1." "we shouldn't be using any metadata or
  anything for our main pipeline, just the VCF and recombination map stuff where the sample labels are names that
  don't mean anything beyond being an identifier."
- **31 Oct** (§4.1, CLAUDE.md). An agent asked the owner a blocking question at about 02:00 while the owner slept, did
  nothing more until the answer at 08:10, and ten paid nodes sat idle for up to six hours. "the tool is fine. Stalling
  over unanswered questions isn't".

## November

- **2 Nov** (§7.4). The Codex runner's `TMPDIR` moved from hpc-work to
  `/rds/project/rds-8b3VcZwY7rY/projects/claude_lab/codex-runner-tmp` when hpc-work's file quota filled.
- **7 Nov** (§2.3). "Even if you change the summation order but get the same result in the end that's fine".
- **8 Nov** (§5.5, §4.8, §5.6). Run the stage tier only where it tests something that changed. Agent instructions travel
  with the work. An agent stamped progress entries 12:55 and 13:05 when the clock read 12:27; earlier entries that night
  carried wrong times too.
- **10 Nov** (§4.8). Blanket bans on Git were too strict: read-only Git is allowed.
- **14 Nov** (§3.3). "we haven't yet released the code to the world or anything".
- **15 Nov** (CLAUDE.md). A session launched 48 replays and ended its turn to wait. 40 finished within the hour; the
  session was woken only when the last straggler ended, 4 hours 45 minutes later, with 40 nodes idle. Later the same day
  it twice answered a question with "starting X now" as a final reply and started nothing.
- **15 Nov** (§4.1). The owner asked why a change made some work serial; the agent answered in a few lines between tool
  calls and carried on; the owner did not see the answer and had to ask again. "when I ask a question you drop
  everything to answer my question, and then pick everything back up".
- **15 Nov** (§4.1). The owner asked for the painting's fixed evidence divisor (10.9) to be replaced by a principled
  method in 4.20. When the first estimators proved unsound, the agent recommended keeping the constant for 4.20 and set
  a 15-minute default to proceed that way. "I ask for a certain thing to be done for a release, then you prototype it
  and figure out that it's more complex than you thought and so you decide to defer it and say it won't go into that
  release and I have to come back and tell you to put it back in. I don't want this to ever happen again".
- **15 Nov** (§4.6). The run calibration of 4.20 gave each chromosome task 19 threads and ran its 27 fits in sequence;
  the scoring of hidden sites, the path labelling and the evidence rescaling were plain Python and NumPy: 71 seconds of
  one-thread work per chromosome, about half the threads idle, measured only after the owner asked why it was slow.
  "We need to parallelize everything by default, only make it single core if there's been testing that shows
  parallelisation doesn't work". "if you find serial bottlenecks in code fix them there and then for the upcoming
  release", then: "Fix only serial bottlenecks for stuff introduced in 4.19 onwards for now. For the future when you
  create something make sure it doesn't cause a serial bottleneck. We'll do a pipeline wide parallelisation pass later".
  In-scope examples from 4.19 and 4.20: the run calibration, the family transmission vote, the pedigree's map-scale
  estimate. Recorded for the later pass (November 2026): family phase, assembly refinement, painting, discovery, kernel
  compilation, refit, release, export, reports, pedigree exclusion.
- **15 Nov** (§4.10). An agent working alone moved between five fronts every few minutes (calibration speed, pedigree
  speed, a failing test, a run hand-over, an early test pass), launched and cancelled runs on unchecked assumptions and
  left finished results unread. "Why are you thrashing around so much and dropping stuff you were supposed to do?" "it
  works on the first thing in the stack/queue etc. and then when the thing is resolved it moves on to the next thing and
  only then to the next thing".
- **16 Nov** (§2.3). An agent replaced two `np.add.at` accumulations with a sort and `np.add.reduceat`, measured it 1.0
  to 1.8 times faster at every size, then reverted it because the gain was small and the last bits changed. "Why are we
  reverting this, it's a speedup and last bits of the sums doesn't matter ... add something to agents.md requiring that
  stuff like this doesn't get reverted".
- **16 Nov** (§3.3). An agent planned to store a contig's cohort genotype frequencies once in the evidence record, then
  reverted to recomputing them in every consumer, only so that later stages could replay runs saved by the previous
  release. "If something is more efficient we should do it even if it breaks previous run replay compatibility".
- **16 Nov** (§5.4). An agent ran each correctness run (a real cross and several simulations) as one job on one node,
  chromosomes queued behind each other, while about 20 paid nodes sat idle for hours. "Why do we have so many idle nodes
  that we aren't doing anything with ... why aren't we fanning out the chrs for that onto different nodes because we're
  not really measuring overall time here, no?"
- **16 Nov** (§2.6). The 4.21 suite failed in 50 places; some were exact float pins that fail on any last-bit change,
  and nobody could tell at a glance which failures were rounding and which science. "make sure that small changes that
  are too small to cause a change in the science don't lead to a failing test"; "future tests must also be robust to
  tiny changes".
- **16 Nov** (§2.6, §4.1). The owner chose to estimate discovery's evidence scale by held-out prediction in a pilot
  (option (b), 06:59). About nine hours went into its machinery (pilot rounds, held-out masks, a forecast, a
  read-sharing guard, a decision rule, stability and sensitivity diagnostics, several advisor reviews) before the check
  of whether the criterion picks what the truth prefers, first called for at 13:54, reported at 16:55: held-out
  prediction picked scale 2 on 10 of 11 simulations where the truth preferred 1 on all of them. Only then, and because
  the owner asked, was the cost of the sound alternative (judging each scale on the whole assembly) worked out:
  +115-125 % pipeline runtime. The owner fixed the scale at 1 ("B5 option (b) is too expensive") and the machinery was
  removed.
  "add any lessons learned for the future to agents.md from the b5 option b thing to minimise risks of something like
  that happening again".
- **16 Nov** (whole file). AGENTS.md had reached 1,401 lines and about 25k tokens, with rules repeated in up to four
  places; the owner asked for it to be compacted ("sleek and maximally efficient") with the incidents moved here.
- **16 Nov** (the cardinal rule). Through the afternoon and evening the coordinator ran B1's prototype, the G1i arms,
  the test re-pins, E6b scoping, docs and a test agent's coordination at once; each was done slowly and badly (the test
  agent waited about 2 hours on missed requests, G1i ran 25-30 node-hours on a superseded upstream, a simple test fix
  ran six hours). The owner: "I think you are focusing on too many things at once and so are doing everything badly
  because of that"; then: "This needs to be the cardinal rule for agents.md".
- **16 Nov** (the cardinal rule, later the same evening). Strict one-at-a-time left nodes idle while a finished item
  waited only on an agent's review. The owner: "this is a bit too strict ... you can move on to the next task if the
  only thing remaining for the previous task is a review being done by a different agent ... This way no more than 2
  tasks are ever active at once and for the earlier you're doing reviews and changing stuff based on the reviews only."
- **16 Nov** (§5.4, non-negotiable 18). At 23:15 the B1 agent ran four independent decision-rule replays as one
  8-core task while 328 of 336 cores were idle, and the coordinator's heartbeat accepted it; AGENTS.md required the
  busy-against-declared check only after a submission. The owner: "Do we not have an agents.md rule that agents should
  always check how many free cores are waiting before submitting a task so they can effectively use them?", then
  "Yes, add such a rule to agents.md".
  Minutes later the same agent submitted two 4-core tasks onto two idle 112-core nodes. The owner: "WHY on 4 cores?
  ... If you submit them, submit on the full damn node if it's idle and you just checked???" The rule was sharpened:
  an idle node gets a whole-node task, and the coordinator checks delegated submissions when they are made.
- **22 Nov** (§1.6, §6.2). "SL4 only" had been read as a rule about running, not requesting: the 4.21 R11 scripts
  refused tasks on the owner's SL2 hosting node, stopping the scoring follower twice. The owner: "Nodes assigned to us
  are nodes assigned to use, we should use them. Doesn't matter if SL2, SL3 or SL4", and "You ask me about SL2/SL3
  when requesting allocations, you don't need to worry about what allocation something runs on for nodes allocated to
  me already". The 17 Oct rule now governs requests only; the account checks were removed from the lab scripts.
- **22 Nov** (§4.1, review proportionate to the change). The EVQ sample-QC speedup had already been shown equivalent where
  it acts (weights equal to rounding, identical TroMau QC decisions, the stall gone, 337 tests passing) when the
  coordinator launched an xhigh Opus review that ran over 30 minutes. The owner: "I don't want to waste tokens given
  something works. Don't stop stuff just now, but keep it in mind for the future."
- **22 Nov** (§4.1, scope). SIM's realistic long reads exposed a pedigree loss on famhs (66 of 98 edges), and the
  coordinator split its attribution off as a separate item. The owner: "we should be doing that with SIM, or at least
  figuring out where the issue is coming from so it can be fixed at its proper stage".
- **22 Nov** (§4.1, questions). Asked "Why is SIM and EVQ taking so long?", the coordinator narrowed SIM's scope and
  split off its long-read attribution without asking; the owner reversed it. Later the owner prefaced a question with
  "Don't change stuff, just answer my question", and ruled: "don't change anything just off of the basis of questions I
  ask unless it's obvious I'm asking for a change rhetorically. If something should change, best to confirm with me."
- **22 Nov** (§2.6, gates). The R4 sampler check froze its pass rule before results. The first fast sampler left one
  near-tie M0 call (0.15 ± 0.12 nats) unreleased in 2 of 3 seeds, a probe showed it was Monte Carlo noise in the
  target both samplers share, and the agent still recorded a fail that the coordinator relayed. The owner: "this is
  obviously an accept, no need to even ask me, we shouldn't be ultra rigid about passing preset gates".
- **22 Nov** (§4.1 and CLAUDE.md, answers once). The rule to repeat an answer at the top of the final reply, meant for
  terminals where interim text scrolls away, made the coordinator write most answers twice in the desktop app, which
  shows interim text. The owner: "I'm seeing various answers twice ... I fear I'm double burning tokens"; they approved
  removing the rule.
- **24 Nov** (§1.6, §6.3, SL2/SL3 cap removed). Compute ran short during the 4.23 LRP build (SL4 workers expiring, SL4
  requests pending on priority). The owner asked for four 112-core sapphire SL3 workers for 12 hours; the coordinator
  submitted two and asked about the rest because four would exceed the 448-CPU SL2 + SL3 cap. The owner: "No, ignore the
  cap, actually remove the 448 CPU cap in agents.md". SL2/SL3 requests remain the owner's to ask for, in the size asked.
- **26 Nov** (§4.6, §1.18, tests and lab work parallel). Diagnosing the 4.23 foreign-share discontinuity, the
  coordinator ran a lab classification script as a single-threaded Python loop over 60 samples x 13,128 bins on one
  contig in a 12-core task, then a half-vectorised version at 8 cores, while whole paid nodes sat idle; earlier the
  release runner had capped each arm at 2 active tasks and run per-contig steps at 0.1-0.4 core use on 76-core nodes.
  The owner: "vectorise everything properly, even for your tests", "why are you submitting over 8 cores when we have
  whole massive nodes idle???", and "Add something to agents.md that all tests must be properly parallelized as well
  for the future".
