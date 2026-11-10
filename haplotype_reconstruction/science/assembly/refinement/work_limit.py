"""Whole-component stop records for shape-bounded window work.

Owns whole-template work receipts and component-stop domains.
Must not choose K, read storage or configure runtime.
Method: docs/methods/founder_refinement.md#work-account (RS07).


The numerical solver owns credit/debits. This layer records its completed window
receipt once, and identifies the entire unfinished conditional search domain.
It does not choose a coefficient or label unperformed candidates as tied evidence.
"""

from __future__ import annotations

import math
import threading
from dataclasses import asdict

from haplotype_reconstruction.contracts.assembly import (
    FitOrdinalBinding,
    LogicalWindowReceipt,
    PhysicalWindowWork,
    WindowAccount,
    WindowDomain,
    WindowOrdinal,
    WindowPathAlphabet,
    WindowRangeAlphabet,
    WindowStop,
    WindowWorkCounters,
    WindowWorkShape,
)
from haplotype_reconstruction.core.errors import InvariantError, ResourceError


class WindowWorkLimit(ResourceError):
    """A numerical window exhausted its account; no component result is usable.

    The sweep attaches completed domains and work after its team has joined. These
    plain values survive process exception transport; no lock/workspace is retained.
    """

    def __init__(self, message):
        super().__init__(message)
        self.domains = ()
        self.work = {}
        self.physical_work = {}
        self.logical_prefix = ()
        self.receipts = ()


class WindowWorkState:
    """Shared by directions and shallow count branches of one component attempt."""

    def __init__(self, source_start=0):
        self.source_start = int(source_start)
        self._lock = threading.Lock()
        self._bound = None
        self._planned = set()
        self._receipts = {}
        self._open_fits = set()
        self._fits = {}
        self._physical = set()

    def bind_fit(self, ordinal, label, panel, kind, branch):
        """Bind a complete fit identity before its worker can run."""
        binding = FitOrdinalBinding(
            int(ordinal), str(label), tuple(tuple(map(int, row)) for row in panel), str(kind), branch
        )
        with self._lock:
            if ordinal in self._fits and self._fits[ordinal] != binding:
                raise InvariantError("fit ordinal was rebound to different inputs")
            self._fits[ordinal] = binding

    def begin_fit(self, ordinal):
        """Open an already bound fit scope before dispatch; close it even on failure."""
        with self._lock:
            self._open_fits.add(int(ordinal))

    def end_fit(self, ordinal):
        """Seal a fit after all its directions and their template workers have joined."""
        with self._lock:
            self._open_fits.remove(int(ordinal))

    def bind(self, prefix, templates):
        """Bind original template indices in one (fit,direction,sweep) scope.

        The caller binds every active template before dispatch, including memo
        lookups; a memo hit completes with zero new work. No result may arrive
        before binding. The returned map owns immutable ordinal records.
        """
        values = {int(i): WindowOrdinal(*map(int, prefix), int(i)) for i in templates}
        with self._lock:
            if any(key in self._planned for key in values.values()):
                raise InvariantError("window-work scope was bound twice")
            self._planned.update(values.values())
        return values

    def reused(self, ordinal, solved=None):
        """Retain the logical work account; a memo hit adds no physical work.

        Every dispatched problem has the same logical account whether obtained by
        solving or exact reuse. Cache scheduling therefore cannot change the prefix.
        """
        with self._lock:
            if ordinal not in self._planned or ordinal in self._receipts:
                raise InvariantError("unbound or duplicate window-work receipt")
            diagnostic = None if not solved else dict(solved[0][-1])
            self._receipts[ordinal] = (diagnostic, None)

    def check(self, ordinal=None):
        """Prune only work above a known failure; lower semantic work must finish."""
        with self._lock:
            if ordinal is not None and ordinal not in self._planned:
                raise InvariantError("window solve has no bound ordinal")
            if self._bound is not None and (ordinal is None or ordinal > self._bound):
                raise WindowWorkLimit("window work lies after the provisional logical stop")

    def finish(self, solved, panel, kind, spans, sizes, alphabets, convention):
        """Record a standalone solver invocation (no concurrent driver)."""
        return self._finish(None, solved, panel, kind, spans, sizes, alphabets, convention)

    def finish_bound(self, ordinal, solved, panel, kind, spans, sizes, alphabets, convention):
        """Record one bound template's cumulative work account and whole domain."""
        return self._finish(ordinal, solved, panel, kind, spans, sizes, alphabets, convention)

    def _finish(self, ordinal, solved, panel, kind, spans, sizes, alphabets, convention):
        if not solved or "work_budget" not in solved[0][2]:
            if ordinal is not None:
                self.reused(ordinal)
            return
        diagnostic = solved[0][2]
        # Read the final template receipt once (scaled accounts attach it only here).
        stopped = bool(diagnostic.get("work_limit", False))
        if ordinal is None:
            # Direct wrapper witnesses have a single serial, implicit scope.
            with self._lock:
                ordinal = WindowOrdinal(0, 0, 0, len(self._planned))
                self._planned.add(ordinal)
        domain = None
        if stopped:
            absolute = tuple((self.source_start + int(a), self.source_start + int(c)) for a, c in spans)
            source = tuple(tuple(map(int, row)) for row in panel)
            alphabet = (
                WindowRangeAlphabet(encoding="prepared_row_range", sizes=tuple(map(int, sizes)))
                if alphabets is None
                else WindowPathAlphabet(
                    encoding="prepared_row_paths",
                    paths=tuple(tuple(tuple(map(int, row)) for row in rows) for rows in alphabets),
                )
            )
            receipt = WindowAccount(
                **{
                    name: (WindowWorkShape(**value) if name == "work_shape" else value)
                    for name, value in diagnostic.items()
                    if name in ("status", "account_scope") or name.startswith("work_") or name.startswith("phase_work")
                }
            )
            domain = WindowDomain(
                kind=kind,
                block_start=absolute[0][0],
                block_stop=absolute[-1][1],
                source_component_start=self.source_start,
                source_component_stop=self.source_start + panel.shape[1],
                source_panel=source,
                focals=tuple(int(item[0]) for item in solved),
                unit_spans=absolute,
                alphabet=alphabet,
                paths_per_focal=math.prod(map(int, sizes)),
                convention_reversed=bool(convention),
                complete_conditional_domain=True,
                evaluated=False,
                meaning="union of single-focal Cartesian products; other source rows fixed",
                account=receipt,
            )
        with self._lock:
            if ordinal not in self._planned or ordinal in self._receipts:
                raise InvariantError("unbound or duplicate window-work receipt")
            self._receipts[ordinal] = (dict(diagnostic), domain)
            self._physical.add(ordinal)
            if stopped:
                self._bound = ordinal if self._bound is None else min(ordinal, self._bound)
        if stopped:
            raise WindowWorkLimit("component window work bound")

    @staticmethod
    def _totals(receipts):
        accounts = [value for value, _ in receipts if value is not None]
        return WindowWorkCounters(
            window_work_limits=sum(bool(v.get("work_limit", False)) for v in accounts),
            window_work_calls=len(accounts),
            window_work_units=sum(int(v["work_used"]) for v in accounts),
            window_work_retries=sum(int(v.get("work_retries", 0)) for v in accounts),
            window_work_discarded_focals=sum(int(v.get("work_discarded", 0)) for v in accounts),
        )

    def snapshot(self):
        """Return canonical domains and logical counters after required workers join.

        The complete prefix includes the failed template's terminal receipt.
        Adaptive earlier fits must be sealed. Physical speculation remains in
        physical_snapshot/receipt_snapshot and never changes scientific budgets.
        """
        with self._lock:
            stop = self._bound
            prefix = sorted(key for key in self._planned if stop is None or key <= stop)
            if any(key not in self._receipts for key in prefix):
                raise InvariantError("window-work logical prefix has unfinished templates")
            if stop is not None and any(fit <= stop.fit for fit in self._open_fits):
                raise InvariantError("window-work logical prefix has unsealed earlier fits")
            domains = () if stop is None else (self._receipts[stop][1],)
            return domains, self._totals([self._receipts[key] for key in prefix])

    def physical_snapshot(self):
        """Return all joined physical work, including speculation past the stop."""
        with self._lock:
            return self._totals([self._receipts[key] for key in self._physical])

    def receipt_snapshot(self):
        """Return bound ordinals and full work accounts, retaining per-account quanta."""
        with self._lock:
            return tuple((key, *self._receipts[key]) for key in sorted(self._receipts))

    @staticmethod
    def account(diagnostic):
        """The cumulative integer work account, with its own scaling quantum."""
        if diagnostic is None:
            return None
        diagnostic = (
            dict(work_phase=0, work_discarded=0, **diagnostic) if "work_phase" not in diagnostic else diagnostic
        )
        return WindowAccount(
            **{
                name: WindowWorkShape(**value) if name == "work_shape" else value
                for name, value in diagnostic.items()
                if name in ("status", "account_scope") or name.startswith("work_") or name.startswith("phase_work")
            }
        )

    def stop_record(self):
        """Seal the completed semantic prefix independently of physical speculation."""
        domains, work = self.snapshot()
        if not domains:
            return None
        with self._lock:
            stop = self._bound
            prefix = tuple(
                LogicalWindowReceipt(key, self.account(self._receipts[key][0]), False)
                for key in sorted(self._planned)
                if key <= stop
            )
            fits = tuple(self._fits[key] for key in sorted(self._fits) if key <= stop.fit)
            physical = PhysicalWindowWork(**asdict(self._totals([self._receipts[key] for key in self._physical])))
        return WindowStop(stop, fits, prefix, work, domains[0], physical)

    def attach(self, error):
        """Attach the completed logical prefix and physical audit to a propagated stop."""
        error.domains, error.work = self.snapshot()
        error.physical_work = self.physical_snapshot()
        error.receipts = self.receipt_snapshot()
        error.logical_prefix = tuple(key for key, _, _ in error.receipts if key <= self._bound)
        return error


__all__ = ["WindowWorkLimit", "WindowWorkState"]
