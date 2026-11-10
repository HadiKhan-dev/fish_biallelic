"""Execute nested children and ordered one-thread maps within CPU and memory claims."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import FIRST_COMPLETED, Future, wait
from typing import Any, TypedDict

from .cpu import (
    CPU_POLL_SECONDS,
    CpuBudget,
    FixedBudget,
    SlotBudget,
    ThreadSlots,
    _no_idle_loans,
    _threading_layer_is_workqueue,
    budget_context,
    executing_task,
    task_owner,
    thread_team,
)
from .memory import adopted, depth_below, held_context, memory_governor

_LOG = logging.getLogger(__name__)


def completed_tasks(budget: CpuBudget, functions, label):
    owner = task_owner()
    governor = memory_governor()
    depth = depth_below(held_context())

    def reserve(index):
        return governor.request(label=label, depth=depth, queued=True)

    if _threading_layer_is_workqueue() or len(functions) <= 1 or (not budget.live and budget.threads() <= 1):
        for index, function in enumerate(functions):
            request = reserve(index)
            request.wait()
            governor.start(request.id)
            try:
                with adopted(request.held), budget_context(budget), thread_team(budget.threads()):
                    value = function(budget)
            finally:
                request.release()
            yield (index, value)
        return
    slots = ThreadSlots(len(functions), budget)

    def execute(index, request):
        slots.start(index)
        governor.start(request.id)
        completed = False
        try:
            own = SlotBudget(slots, index)
            with executing_task(owner), adopted(request.held), budget_context(own), thread_team(own.threads()):
                value = functions[index](own)
            completed = True
            return value
        finally:
            # On fail-fast unwind, descendants may still be in native calls.
            # Keep their CPU token until worker-process termination and reaping.
            if completed:
                slots.finish(index)
            request.release()

    queued = list(range(len(functions)))
    running: dict[Future[Any], int] = {}
    yield from _drive_slots(slots, queued, running, reserve, execute, governor)


def _drive_slots(slots, queued, running, reserve, execute, governor):
    """Start daemon threads only for admitted slots; propagate failure before cleanup.

    Python cannot interrupt another thread's native call. On failure the worker
    sends the exception immediately and its coordinator terminates the process.
    Daemon threads avoid an executor's unconditional interpreter-exit join.
    """
    pending = None

    def perform(future, index, request):
        if not future.set_running_or_notify_cancel():
            slots.finish(index)
            request.release()
            return
        try:
            future.set_result(execute(index, request))
        except BaseException as error:
            future.set_exception(error)

    try:
        while queued or running:
            limit = slots.publish()
            starting = []
            slots.waiting = 0
            while queued and len(running) + len(starting) < limit:
                if not slots.admit(queued[0]):
                    slots.waiting = 1
                    break
                if pending is None:
                    pending = reserve(queued[0])
                if not pending.poll():
                    if running or starting:
                        slots.finish(queued[0])
                        break
                    pending.wait()
                starting.append((queued.pop(0), pending))
                pending = None
            for index, request in starting:
                future: Future[Any] = Future()
                running[future] = index
                threading.Thread(target=perform, args=(future, index, request), daemon=True).start()
            timeout = CPU_POLL_SECONDS
            with governor.suspended():
                done, _ = wait(running, return_when=FIRST_COMPLETED, timeout=timeout)
            for future in done:
                yield (running.pop(future), future.result())
    except BaseException:
        slots.stopped.set()
        if pending is not None:
            pending.release()
        for future in running:
            future.cancel()
        raise


class _ThreadMapState(TypedDict):
    taken: int
    finished: int
    workers: int
    allowed: int
    error: BaseException | None


def one_thread_map(budget: CpuBudget, function, items, label):
    owner = task_owner()
    results = [None] * len(items)
    state: _ThreadMapState = dict(taken=0, finished=0, workers=0, allowed=1, error=None)
    changed = threading.Condition()
    governor = memory_governor()
    caller = held_context()
    depth, parent = (depth_below(caller), caller[0] if caller else 0)

    def call(index):
        return function(items[index])

    def work():
        presence = governor.presence(label, depth, parent=parent)
        try:
            with (
                executing_task(owner),
                thread_team(1),
                budget_context(FixedBudget(1)),
                adopted(presence.held),
                _no_idle_loans(),
            ):
                while True:
                    with changed:
                        if (
                            state["error"] is not None
                            or state["taken"] == len(items)
                            or state["workers"] > state["allowed"]
                        ):
                            state["workers"] -= 1
                            changed.notify()
                            return
                        index = state["taken"]
                        state["taken"] += 1
                    try:
                        results[index] = call(index)
                    except BaseException as error:
                        with changed:
                            if state["error"] is None:
                                state["error"] = error
                    with changed:
                        state["finished"] += 1
                        changed.notify()
        except BaseException as error:
            with changed:
                if state["error"] is None:
                    state["error"] = error
                changed.notify_all()
        finally:
            presence.release(record=False)

    started = []
    with governor.suspended(), changed:
        try:
            while state["finished"] < len(items) and state["error"] is None:
                # changed also guards worker starts/retirement: H cannot grow
                # between this snapshot, the upstream release and publication.
                state["allowed"] = max(1, int(budget.allowance(state["workers"])))
                # Size workers for all unfinished items, including running calls, so queued work
                # can start when the live share grows while a long call is still running.
                while state["workers"] < min(state["allowed"], len(items) - state["finished"]):
                    state["workers"] += 1
                    started.append(threading.Thread(target=work, daemon=True))
                    started[-1].start()
                changed.wait(CPU_POLL_SECONDS)
        except BaseException as error:
            if state["error"] is None:
                state["error"] = error
    if state["error"] is None:
        for worker in started:
            worker.join()
    if state["error"] is not None:
        raise state["error"]
    return results
