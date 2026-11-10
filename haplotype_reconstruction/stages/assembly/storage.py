"""Build refinement storage capabilities from explicit, deterministic task choices.

Evidence residency and message checkpoints are independent routes. Forced choices
exercise exact fallback reads and reverse-clamp reconstruction without environment
flags or live-memory tests. Numerical fill/read kernels remain in science.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import TracebackType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from haplotype_reconstruction.science.assembly.refinement.component_workspace import frozen
from haplotype_reconstruction.science.assembly.refinement.window import (
    COST_ARRAYS,
    PARTIAL_SITE_BYTES,
    WINDOW_BYTES,
    WINDOW_SITE_BYTES,
    _fill_costs,
    _fill_logs,
    refill,
)

__all__ = ["StoragePlan", "Windows"]

# Exact kernel capability: evidence, masks, hazards, log buffers, map, costs, metadata.
WindowBuffers = tuple[
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.int64],
]
WindowView = tuple[*WindowBuffers, bool]


@dataclass(frozen=True)
class StoragePlan:
    """Task storage choices; capacity is in sites, message_step is in blocks.

    None follows the original deterministic dimension rule; explicit choices force
    the route for replay. These fields affect resources only, never scientific caps.
    """

    resident: bool | None = None
    capacity: int | None = None
    message_step: int | None = None


class Windows:
    """The windows of one component (ComponentWorkspace.windows): its resident window, or every consumer's own.

    view(K, reverse) returns the window tuple a kernel reads: the resident window with K's costs in that
    orientation, else the window of the calling thread's open lease of K (`lease`), else a fresh window that
    lives as long as the caller holds it. A consumer that runs many kernels (a search) holds a lease, so they
    share its window and refill it only where they move.

    Every window tuple has one argument type, whichever route built the component (stage or replay): the
    evidence frozen (C-contiguous, read-only, shared), the masks and hazards the windows' own C-contiguous writable
    copies, then the windows' scratch. This is the type the refinement kernels declare and the warm-up compiles.
    """

    def __init__(
        self,
        evidence: NDArray[np.float32],
        kept: NDArray[np.bool_],
        partial: NDArray[np.bool_],
        hazards: NDArray[np.float64],
        *,
        resident: bool,
        capacity: int,
    ) -> None:
        self.evidence = frozen(evidence, np.float32)
        self.kept = np.array(kept, np.bool_, order="C")
        self.partial = np.array(partial, np.bool_, order="C")
        self.hazards = np.array(hazards, np.float64, order="C")
        self.samples, self.sites = int(evidence.shape[0]), int(evidence.shape[1])
        self.resident = bool(resident)
        self.capacity = self.sites if self.resident else min(self.sites, int(capacity))
        # partial sites a window can meet: the most in any capacity-long stretch
        cum = np.concatenate(([0], np.cumsum(self.partial, dtype=np.int64)))
        self.partial_capacity = int(np.max(cum[self.capacity :] - cum[: -self.capacity])) if self.sites else 0
        self._lock = threading.Lock()
        self._local = threading.local()
        self._logs: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]] | None = None
        self._costs: dict[tuple[int, bool], WindowView] = {}

    # ---------------------------------------------------------------- resident
    def _resident_logs(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        with self._lock:
            if self._logs is None:
                logs = np.zeros((self.samples, self.sites, 3))
                plogs = np.zeros((self.samples, max(1, self.partial_capacity), 4))
                pmap = np.full(self.sites, -1, np.int64)
                pmap[self.partial] = np.arange(int(self.partial.sum()))
                _fill_logs(self.evidence, self.kept, self.partial, logs, plogs, pmap, 0, 0, self.samples, 0, self.sites)
                self._logs = (logs, plogs, pmap)
            return self._logs

    def _resident(self, founders: int, reverse: bool) -> WindowView:
        key = (int(founders), bool(reverse))
        logs, plogs, pmap = self._resident_logs()
        with self._lock:
            found = self._costs.pop(key, None)
            if found is None:
                costs = np.empty((self.samples, self.sites), np.float32)
                _fill_costs(
                    self.evidence,
                    self.kept,
                    self.hazards,
                    costs,
                    0,
                    0,
                    self.samples,
                    0,
                    self.sites,
                    int(founders),
                    bool(reverse),
                )
                meta = np.array([0, self.sites, 0, self.samples, int(founders), int(reverse), self.sites, 1], np.int64)
                found = (
                    self.evidence,
                    self.kept,
                    self.partial,
                    self.hazards,
                    logs,
                    plogs,
                    pmap,
                    costs,
                    meta,
                    bool(reverse),
                )
            self._costs[key] = found
            while len(self._costs) > COST_ARRAYS:
                self._costs.pop(next(iter(self._costs)))
            return found

    # ---------------------------------------------------------------- streaming
    def _fresh(self, founders: int) -> WindowBuffers:
        """Empty buffers of a streaming window of K = founders (nothing held)."""
        rows, size = self.samples, self.capacity
        logs = np.empty((rows, size, 3))
        plogs = np.empty((rows, max(1, self.partial_capacity), 4))
        pmap = np.full(size, -1, np.int64)
        costs = np.empty((rows, size), np.float32)
        meta = np.array([0, 0, 0, 0, int(founders), -1, self.sites, 0], np.int64)
        return (self.evidence, self.kept, self.partial, self.hazards, logs, plogs, pmap, costs, meta)

    def window_bytes(self, founders: int | None = None) -> int:
        """Bytes of one consumer's window (0 for a resident component: its tables are the workspace's)."""
        if self.resident:
            return 0
        return (
            self.samples * self.capacity * WINDOW_SITE_BYTES
            + self.capacity * 8
            + self.samples * max(1, self.partial_capacity) * PARTIAL_SITE_BYTES
        )

    def view(self, founders: int, reverse: bool = False) -> WindowView:
        """The window tuple of K = founders in the given orientation (class docstring)."""
        if self.resident:
            return self._resident(founders, reverse)
        leases = getattr(self._local, "leases", None)
        buffers = None if not leases else leases.get(int(founders))
        if buffers is None:
            buffers = self._fresh(founders)
        return (*buffers, bool(reverse))

    def sample_slices(self, s0: int, s1: int) -> list[tuple[int, int]]:
        """Consecutive slices of samples [s0, s1) whose windows over every site take about WINDOW_BYTES of codes
        0-2 and costs (28 bytes per sample-site; the component's partial sites add 32 bytes per sample and partial
        site) (one slice for a resident component): for kernels that revisit the whole component per sample."""
        if self.resident:
            return [(s0, s1)]
        size = max(1, WINDOW_BYTES // max(1, WINDOW_SITE_BYTES * self.sites))
        return [(a, min(s1, a + size)) for a in range(s0, s1, size)]

    def samples_view(self, founders: int, s0: int, s1: int, reverse: bool = False) -> WindowView:
        """A window holding every site of samples [s0, s1) with K = founders (sample_slices), filled now; the
        resident window for a resident component."""
        if self.resident:
            return self._resident(founders, reverse)
        rows, size = s1 - s0, self.sites
        partials = int(self.partial.sum())
        win = (
            self.evidence,
            self.kept,
            self.partial,
            self.hazards,
            np.empty((rows, size, 3)),
            np.empty((rows, max(1, partials), 4)),
            np.full(size, -1, np.int64),
            np.empty((rows, size), np.float32),
            np.array([0, 0, 0, 0, int(founders), -1, self.sites, 0], np.int64),
            bool(reverse),
        )
        refill(win, 0, self.sites, s0, s1)
        return win

    def lease(self, founders: int) -> _Lease:
        """Context manager: the calling thread's kernels of K = founders share one streaming window until it
        closes (nothing for a resident component). Nested leases of the same K reuse the open one."""
        return _Lease(self, int(founders))


class _Lease:
    def __init__(self, windows: Windows, founders: int) -> None:
        self.windows = windows
        self.founders = founders
        self.opened = False

    def __enter__(self) -> _Lease:
        windows = self.windows
        if windows.resident:
            return self
        leases = getattr(windows._local, "leases", None)
        if leases is None:
            leases = windows._local.leases = {}
        if self.founders not in leases:
            leases[self.founders] = windows._fresh(self.founders)
            self.opened = True
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> Literal[False]:
        if self.opened:
            self.windows._local.leases.pop(self.founders, None)
        return False
