"""Assign contig owners to lanes and rebalance their CPU leases."""

from __future__ import annotations

import logging
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.errors import SettingsError
from .cpu import LaneLease, LeaseState, PoolCounters, read_counter
from .graph import TaskKey

_LOG = logging.getLogger(__name__)


CONTIG_LANE_CORES = 16
STRAGGLER_RATIO = 2.0
# At most this share of a round's declared work runs on minimum teams (minimum_teams): a team holds its CPUs through
# its serial sections (66-82% lane efficiency at five threads on pearly chr3 l2 blocks), so the share bounds what a
# crowded round can spend on teams, including rounds whose cost model overstates the spread of unit work.
TEAM_WORK_SHARE = 0.125


def minimum_teams(costs: Sequence[float], widths: Sequence[int], cores: int) -> tuple[int, ...]:
    """Minimum CPU teams for the stragglers of one round of independent units, from their declared costs alone.

    A unit whose declared cost is at least STRAGGLER_RATIO times the round's upper-median positive cost is a
    straggler (the contig lanes' straggler definition). On a node crowded with its peers a unit starts on one CPU and
    widens only as they finish, so a straggler would run several times longer than the round's other units.
    Stragglers, heaviest first (ties to the lower index), take a minimum team of their useful width ``widths[i]``
    (the CPUs their kernels use at full grain), capped at one contig lane's planned CPUs (CONTIG_LANE_CORES) and at
    half of ``cores``: a lane admits a team within its equal share, and the round's other units keep CPUs. A
    straggler whose cost is at least the round's cost per CPU (total / cores) is critical: on one CPU it would outlast
    the balanced round, so it always takes its team. The other stragglers stop before their teamed declared cost would
    exceed TEAM_WORK_SHARE of the round's. Every other unit's minimum is one CPU. A team is a CPU minimum for
    scheduling only; it never changes a result.
    """
    count = len(costs)
    if len(widths) != count:
        raise SettingsError("minimum_teams needs one width per cost")
    teams = [1] * count
    cap = min(CONTIG_LANE_CORES, max(1, int(cores) // 2))
    values = [max(0.0, float(value)) for value in costs]
    positive = sorted(value for value in values if value > 0.0)
    if cap < 2 or not positive:
        return tuple(teams)
    total = sum(values)
    threshold = STRAGGLER_RATIO * positive[len(positive) // 2]
    allowance = TEAM_WORK_SHARE * total
    teamed = 0.0
    for index in sorted(range(count), key=lambda i: (-values[i], i)):
        value = values[index]
        critical = value * int(cores) >= total
        if value < threshold or (not critical and teamed + value > allowance):
            break
        width = min(cap, int(widths[index]))
        if width >= 2:
            teams[index] = width
            teamed += 0.0 if critical else value
    return tuple(teams)


@dataclass(frozen=True)
class ChainPlan:
    """One scheduling chain's cost and last boundary, independent of model edges."""

    affinity: str
    weight: float
    tasks: tuple[TaskKey, ...]


class LaneAuthority:
    """Native leases and group shares, with nonblocking coordinator observations.

    Workers may die while holding CPU locks. The coordinator never waits for
    those locks: it returns to transport liveness checks and aborts the group.
    Only the worker's own boundary changes its running team; every lease floor
    covers all currently claimed group CPUs.
    """

    def __init__(self, context: Any, cores: int, lanes: int, processes: int, chains: Sequence[ChainPlan]):
        self.chains = tuple(chains)
        self.indices = {chain.affinity: i for i, chain in enumerate(chains)}
        self.state = LeaseState(
            context,
            lanes,
            cores,
            task_weights=[chain.weight for chain in chains] if chains else None,
            straggler_ratio=STRAGGLER_RATIO,
        )
        self.seats = [LaneLease(self.state, i) for i in range(lanes)]
        self.groups = tuple(PoolCounters(context, 1, processes) for _ in range(lanes))
        for i, group in enumerate(self.groups):
            group.lease_state, group.lease_index = self.state, i
            group.start_batch(processes, processes)
        self.owners: dict[str | None, int] = {}
        self.finished: set[str] = set()

    def acquire(self, owner: str | None, *, minimum: int = 1) -> int | None:
        """``owner``'s lane once it holds ``minimum`` CPUs (None meanwhile). A lane that already holds them returns
        at once: the dispatcher refreshes every lane's total once per pass (a pass walks thousands of ready
        blocks), not once per task."""
        if owner in self.owners:
            index = self.owners[owner]
            if not self.state.lock.acquire(False):
                return None
            try:
                if self.state.lent[index] == 0 and self.state.claims[index] >= minimum:
                    return index
            finally:
                self.state.lock.release()
        if owner not in self.owners:
            if not self.state.lock.acquire(False):
                return None
            try:
                if not self._register(owner):
                    return None
            finally:
                self.state.lock.release()
        index = self.owners[owner]
        self.refresh()
        if not self.state.lock.acquire(False):
            return None
        try:
            claim = int(self.state.claims[index])
            free = self.state.free()
            if claim < minimum:
                if free < minimum - claim:
                    return None
                self.state.claims[index] = minimum
            self.state.lent[index] = 0
        finally:
            self.state.lock.release()
        self.refresh()
        return index

    def _register(self, owner: str | None) -> bool:
        """Give ``owner`` a free lane, acquiring (counted in every lane's share before it claims a CPU); under the
        state lock. False when no lane is free."""
        free = [i for i in range(len(self.groups)) if i not in self.owners.values()]
        if not free:
            return False
        index = free[0]
        self.owners[owner] = index
        self.state.lent[index] = -1
        self.state.wants[index] = 0  # a new owner starts unlimited (limit() sets its want from its own tasks)
        if owner in self.indices and owner not in self.finished:
            weight = self.state.task_weights[self.indices[owner]]
            self.state.weights[index] = weight if weight >= self.state.threshold else 0.0
        return True

    def announce(self, owners: Iterable[str | None]) -> None:
        """Register owners with ready work as acquiring lanes before any claims CPUs in this pass,
        so each lane's initial share accounts for its ready peers."""
        if not self.state.lock.acquire(False):
            return
        try:
            for owner in owners:
                if owner not in self.owners and not self._register(owner):
                    break
        finally:
            self.state.lock.release()

    def refresh(self) -> None:
        for index in sorted(self.owners.values()):
            group = self.groups[index]
            if not group.admission.acquire(False):
                continue
            try:
                if not self.state.lock.acquire(False):
                    continue
                try:
                    if self.state.claims[index] <= 0 or self.state.lent[index] == 1:
                        continue
                    _, total = self.seats[index]._grow_locked(read_counter(group.claimed))
                    raised = total > read_counter(group.total)
                    group.total.get_obj().value = total
                    if raised:
                        group.admission.notify_all()  # tasks waiting in begin_task start on the new CPUs
                finally:
                    self.state.lock.release()
            finally:
                group.admission.release()

    def limit(self, wants: Mapping[str | None, int]) -> None:
        """Per owner, the most CPUs its lane can use now (0: no limit). A lane that has nothing ready to start is
        limited to its running tasks' maxima, so the rest of its share is free for lanes with ready work (a contig's
        context assembly between rounds beside contigs with thousands of ready blocks); the limit lifts once it has
        ready work again, and borrowers return CPUs at their next boundary. Long tasks re-read
        their share within a step."""
        if not self.state.lock.acquire(False):
            return
        try:
            for owner, index in self.owners.items():
                want = max(0, int(wants.get(owner, 0)))
                # A lane with tasks on their way (sent, or expected by a free worker) is not limited: they take
                # their shares when they begin. Preserve room for their initial claim even when a task
                # was dispatched holding only one CPU and has no later boundary within the step.
                if want and self.groups[index].arriving():
                    want = 0
                self.state.wants[index] = want
        finally:
            self.state.lock.release()

    def lend_idle(self, owner: str | None) -> None:
        if owner not in self.owners:
            return
        index = self.owners[owner]
        group = self.groups[index]
        if not group.admission.acquire(False):
            return
        try:
            if not self.state.lock.acquire(False):
                return
            try:
                if not read_counter(group.claimed) and self.state.claims[index] > 0:
                    self.state.claims[index] = 1
                    self.state.held[index] = 0
                    self.state.lent[index] = 1
                    group.total.get_obj().value = 1
            finally:
                self.state.lock.release()
        finally:
            group.admission.release()
        self.refresh()

    def reconcile(self, needed: set[str | None], results: Container[TaskKey]) -> None:
        if not self.state.lock.acquire(False):
            return
        try:
            for chain in self.chains:
                if chain.affinity not in self.finished and all(key in results for key in chain.tasks):
                    index = self.owners.get(chain.affinity)
                    self.state.remaining.value = max(0.0, self.state.remaining.value - chain.weight)
                    if index is not None:
                        self.state.weights[index] = 0.0
                    self.finished.add(chain.affinity)
            for owner, index in tuple(self.owners.items()):
                if owner not in needed and not read_counter(self.groups[index].claimed):
                    # Clear demand as well as claims so the next owner starts without its predecessor's limit.
                    self.state.claims[index] = self.state.held[index] = self.state.lent[index] = 0
                    self.state.wants[index] = 0
                    if self.state.proportional:
                        self.state.weights[index] = 0.0
                    del self.owners[owner]
        finally:
            self.state.lock.release()
        self.refresh()


def partition_chains(weights: Sequence[float], workers: int) -> tuple[tuple[int, ...], ...]:
    """ENG-22 static placement: cost, then load/task-count/lower-index ties."""
    if workers < 1:
        raise SettingsError("chain partitions require a positive worker count")
    partitions: list[list[int]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for index in sorted(range(len(weights)), key=lambda i: (-weights[i], i)):
        worker = min(range(workers), key=lambda i: (loads[i], len(partitions[i]), i))
        partitions[worker].append(index)
        loads[worker] += max(0.0, weights[index])
    return tuple(tuple(partition) for partition in partitions)
