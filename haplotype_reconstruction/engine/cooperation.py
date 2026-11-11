"""Deferred checkpoint calls and coordinator lifetime for cooperative graph execution."""

from __future__ import annotations

from dataclasses import dataclass

POLL_SECONDS = 5.0
HEARTBEAT_SECONDS = 30.0


@dataclass(frozen=True)
class Deferred:
    """No result yet: retry this call without satisfying any graph dependency."""

    reason: str


_WATCHING_PARENT = False


def watch_coordinator() -> None:
    """Release a worker's claims when its coordinating process dies.

    Forkserver workers have the forkserver as their OS parent; multiprocessing's
    parent sentinel instead tracks the actual coordinator, without PID polling.
    """
    global _WATCHING_PARENT
    if _WATCHING_PARENT:
        return
    from multiprocessing import parent_process
    from multiprocessing.connection import wait
    from threading import Thread

    parent = parent_process()
    if parent is None:
        return  # In-process execution: the claim holder is the coordinator itself.
    _WATCHING_PARENT = True

    def lost_coordinator():
        import os

        wait([parent.sentinel])
        os._exit(1)  # Only this orphan worker; kernel flock releases all its claims.

    Thread(target=lost_coordinator, name="coordinator-parent", daemon=True).start()
