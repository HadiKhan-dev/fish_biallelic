# Claude Code

@AGENTS.md

`AGENTS.md` is the single source of shared policy for every agent, Claude Code included. This file adds only what is
specific to Claude Code. Shared rules go in `AGENTS.md`, never here.

- **Answer the owner's questions in a reply of their own.** When the owner asks a question during a task, answer it
  fully in its own reply, then continue the work. Text written between tool calls is easily missed.
- **Claude subagents run at xhigh effort, not max** (owner, 22 October 2026). Set the effort explicitly on every
  subagent you create.
