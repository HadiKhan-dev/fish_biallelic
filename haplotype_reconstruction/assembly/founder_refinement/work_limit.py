"""Whole-component stop records for shape-bounded window work.

The numerical solver owns credit/debits. This layer records its completed window
receipt once, and identifies the entire unfinished conditional search domain.
It does not choose a coefficient or label unperformed candidates as tied evidence.
"""
from __future__ import annotations

import math
import threading


class WindowWorkLimit(RuntimeError):
    """A numerical window exhausted its account; no component result is usable."""


class WindowWorkState:
    """Shared by directions and shallow count branches of one component attempt."""

    def __init__(self, source_start=0):
        self.source_start = int(source_start)
        self._lock = threading.Lock()
        self._stopped = False
        self._domains = []
        self._calls = self._used = self._retries = self._discarded = 0
        self._seconds = 0.0

    def check(self):
        # Active calls finish under their own finite account. New calls abort;
        # execution.thread_map/concurrent join active threads before propagation.
        with self._lock:
            if self._stopped:
                raise WindowWorkLimit("component window work bound")

    def finish(self, solved, panel, kind, spans, sizes, alphabets, convention, seconds):
        if not solved or 'work_budget' not in solved[0][2]:
            return
        diagnostic = solved[0][2]
        # Read the final template receipt once (scaled accounts attach it only here).
        stopped = bool(diagnostic.get('work_limit', False))
        with self._lock:
            self._calls += 1
            self._used += int(diagnostic['work_used'])
            self._retries += int(diagnostic.get('work_retries', 0))
            self._discarded += int(diagnostic.get('work_discarded', 0))
            self._seconds += float(seconds)
        if not stopped:
            return
        absolute = tuple((self.source_start + int(a), self.source_start + int(c)) for a, c in spans)
        source = tuple(tuple(map(int, row)) for row in panel)
        alphabet = (dict(encoding="prepared_row_range", sizes=tuple(map(int, sizes)))
                    if alphabets is None else dict(encoding="prepared_row_paths",
                    paths=tuple(tuple(tuple(map(int, row)) for row in rows) for rows in alphabets)))
        receipt = {key: value for key, value in diagnostic.items()
                   if key in ('status', 'account_scope') or key.startswith('work_') or key.startswith('phase_work')}
        domain = dict(kind=kind, block_start=absolute[0][0], block_stop=absolute[-1][1],
            source_component_start=self.source_start, source_component_stop=self.source_start + panel.shape[1],
            source_panel=source, focals=tuple(int(item[0]) for item in solved), unit_spans=absolute,
            alphabet=alphabet, paths_per_focal=math.prod(map(int, sizes)),
            convention_reversed=bool(convention), complete_conditional_domain=True, evaluated=False,
            meaning="union of single-focal Cartesian products; other source rows fixed",
            account=receipt)
        with self._lock:
            self._domains.append(domain)
            self._stopped = True
        raise WindowWorkLimit("component window work bound")

    def snapshot(self):
        # Called after execution contexts have joined all active threads.
        with self._lock:
            domains = tuple(sorted(self._domains, key=lambda d: repr((d['kind'], d['unit_spans'],
                d['source_panel'], d['focals'], d['convention_reversed']))))
            return domains, dict(window_work_limits=len(domains), window_work_calls=self._calls,
                window_work_units=self._used, window_work_retries=self._retries,
                window_work_discarded_focals=self._discarded), self._seconds
