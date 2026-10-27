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
- **No end-to-end runs to check a fix** (owner, 26 October 2026; shared rule in `AGENTS.md` §2.6 and §5.5).
  - Validate each fix where it acts: its fixture, replays of the affected blocks or components on saved inputs, and
    competing outputs scored on common inputs.
  - The single end-to-end run is for the final integrated release candidate. A downstream issue can make a correct
    fix look worse end to end, and those runs are slow.
