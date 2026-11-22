# Claude Code

@AGENTS.md

`AGENTS.md` holds every shared rule; this file adds only what is specific to Claude Code. Date tags refer to
`AGENTS_HISTORY.md`.

- **Answers are text, not a turn's end.** Answer an owner's question as text before any further tool call (AGENTS
  §4.1), then keep calling tools in the same turn. Text written between tool calls is easily missed: repeat the answer
  at the top of the final reply.
- **A final reply ends the turn; nothing happens until something wakes the session** [15 Nov]. While work remains,
  never end a turn on a wait or an answer. End it only when the work is done or needs a decision only the owner can
  make. Never write "starting X now" or "next I will" in a final reply: start X, then report it running.
- **Never wait idle for a batch.** Work on the item's own other steps while it runs (never another item: the cardinal
  rule in AGENTS.md), collect results as they arrive and decide on partial results. Every wait has a wake-up that does
  not depend on the slowest task: a background waiter with a deadline and a partial-completion condition ("40 of 48
  done, or 30 minutes") plus a heartbeat. If a turn must end with work running, say what will wake the session and
  when.
- **AskUserQuestion blocks the session with no timeout** [31 Oct]. Use it only after launching everything that can
  proceed, when the owner is evidently present (replied within minutes) and no running work will need you before they
  can answer. Otherwise ask in a normal reply with the recommended option and the 15-minute default, start a background
  timer (Bash `sleep 900` with `run_in_background`), keep working, and apply the default when it fires.
- **Subagents run at effort xhigh, not max** [22 Oct], set explicitly on every subagent.
- **Act on the idle-node reminder** [29 Oct]. A user-level hook (`~/.claude/settings.json` running
  `claude_lab/nodeq/hook.sh`) adds an `[idle nodes]` line once workers have idled 10 minutes. Act on it in the same turn
  (AGENTS §5.4): queue useful authorized work or say why none is ready. If your own tasks cause it (serial loops over
  units, few of their declared cores busy), fan them out. Before ending a turn with work queued, check `nodeq status`
  once: busy cores near what your tasks declared, no idle nodes that per-unit tasks could fill.
