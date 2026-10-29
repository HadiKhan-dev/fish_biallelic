# Claude Code

@AGENTS.md

`AGENTS.md` is the single source of shared policy for every agent, Claude Code included. This file adds only what is
specific to Claude Code. Shared rules go in `AGENTS.md`, never here.

- **Answer the owner's questions in a reply of their own.** When the owner asks a question during a task, answer it
  fully in its own reply, then continue the work. Text written between tool calls is easily missed.
- **Claude subagents run at xhigh effort, not max** (owner, 22 October 2026). Set the effort explicitly on every
  subagent you create.
- **Asking a reviewer never replaces your own work on the problem** (owner, 26 October 2026).
  - When Astra or another agent reviews or investigates something, work on the same problem yourself, in parallel:
    analyse it, test hypotheses and try to solve it.
  - Then compare conclusions. A review checks your reasoning; it never substitutes for it.
  - Never wait idle for a reviewer's verdict on a question you could be investigating.
- **Act on the idle-node reminder** (owner, 29 October 2026).
  - A user-level hook (`~/.claude/settings.json`, running `claude_lab/nodeq/hook.sh`) adds an `[idle nodes]` line to
    the context. It fires once worker nodes have been idle for 10 minutes or more.
  - Act on it in the same turn (`AGENTS.md` §5.4): queue useful authorized work through `nodeq`, or say why none is
    ready.
  - It also fires while your own tasks run. If they loop over independent units (chromosomes, samples, variants) or
    use few of their declared cores, they are the cause. Fan them out per unit and parallelise inside each task
    (`AGENTS.md` §5.4) instead of reporting that nothing is ready.
  - Before ending a turn with work queued, check `nodeq status` once: busy cores should be close to what your tasks
    declared, with no idle nodes that per-unit tasks could fill.
- **No end-to-end runs to check a fix** (owner, 26 October 2026; shared rule in `AGENTS.md` §2.6 and §5.5).
  - Validate each fix where it acts: its fixture, replays of the affected blocks or components on saved inputs, and
    competing outputs scored on common inputs.
  - The single end-to-end run is for the final integrated release candidate. A downstream issue can make a correct
    fix look worse end to end, and those runs are slow.
