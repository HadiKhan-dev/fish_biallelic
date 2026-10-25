"""Owner-only, lock-free detection of a stalled memory ledger.

This is a last-resort run failure, not an admission rule. A frozen ledger with waiters must
also have used less than one CPU core over the entire observation window. Process turnover
restarts that window: CPU from exited workers must never be mistaken for inactivity.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import threading
import time

from . import config


def process_tree(root: int) -> dict:
    """Read this run's process tree, including children created by non-main threads.

    Keys include Linux start ticks, so PID reuse cannot confuse CPU differences or shutdown.
    An unreadable live process invalidates the sample instead of implying zero CPU.
    """
    pending, found, seen = [root], {}, set()
    ticks = os.sysconf("SC_CLK_TCK")
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
            fields = text.rpartition(")")[2].split()
            row = dict(pid=pid, started=int(fields[19]), parent=int(fields[1]),
                       name=text.partition("(")[2].rpartition(")")[0], state=fields[0],
                       cpu_seconds=(int(fields[11]) + int(fields[12])) / ticks)
            found[pid, row["started"]] = row
            for task in Path(f"/proc/{pid}/task").iterdir():
                try:
                    pending.extend(map(int, (task / "children").read_text().split()))
                except FileNotFoundError:     # thread exited during this sample
                    continue
        except (FileNotFoundError, ProcessLookupError):
            continue                         # process exited during this sample
    return found


class LedgerWatchdog:
    """One daemon in the ledger owner; attachers never create a monitor.

    The optional shorter intervals are for bounded runtime checks. No scientific option,
    environment variable, worker, or additional CPU budget is introduced.
    """

    def __init__(self, governor, *, stall_seconds=None, poll_seconds=None, diagnostic_directory=None):
        self.governor = governor
        self.pid = os.getpid()
        self.stall_seconds = (60 * config.MEMORY_STALL_MINUTES if stall_seconds is None else stall_seconds)
        self.poll_seconds = config.MEMORY_STALL_POLL_SECONDS if poll_seconds is None else poll_seconds
        self.diagnostic_directory = (Path(diagnostic_directory) if diagnostic_directory is not None
                                     else Path(governor.handle).parent)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="memory-ledger-watchdog", daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if os.getpid() == self.pid and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1.0)

    def _run(self):
        from .memory import _FIELDS, _WAITING
        # Pressure-based ledgers also count starts separately from grants (requests).
        progress_indices = tuple(_FIELDS.index(name) for name in ("requests", "released", "starts")
                                 if name in _FIELDS)
        baseline = None
        while not self.stop.wait(self.poll_seconds):
            # Same aligned int64 loads as MemoryGovernor._released(); never acquire ledger.locked().
            h = self.governor._ledger.h
            counters = tuple(int(h[i]) for i in progress_indices)
            waiting = int(h[_WAITING])
            if waiting <= 0:
                baseline = None
                continue
            try:
                processes = process_tree(self.pid)
            except (OSError, ValueError, IndexError):
                baseline = None              # unavailable CPU evidence is not evidence of a stall
                continue
            now = time.monotonic()
            if (not processes or baseline is None or counters != baseline[0]
                    or processes.keys() != baseline[2].keys()):
                baseline = counters, now, processes
                continue
            elapsed = now - baseline[1]
            if elapsed < self.stall_seconds:
                continue
            cpu = []
            for key, row in processes.items():
                seconds = max(0.0, row["cpu_seconds"] - baseline[2][key]["cpu_seconds"])
                cpu.append(dict(row, window_cpu_seconds=seconds, window_cores=seconds / elapsed))
            cores = sum(row["window_cores"] for row in cpu)
            if cores >= config.MEMORY_STALL_CPU_CORES:
                baseline = counters, now, processes
                continue
            # Recheck after reading /proc: admission progress during the sample cancels the diagnosis.
            if (tuple(int(h[i]) for i in progress_indices) != counters or int(h[_WAITING]) <= 0
                    or self.stop.is_set()):
                baseline = None
                continue
            try:
                self._fail(elapsed, cores, cpu, processes)
            finally:
                os._exit(1)                  # diagnostic failure must not leave the run hung

    def _fail(self, elapsed, cores, cpu, processes):
        """Write the snapshot, then stop only verified descendants and the run owner.

        Do not unwind pool/ledger finalizers: they may need the very lock that stopped the run.
        Existing checkpoints are untouched. In-flight units remain incomplete and can be resumed.
        """
        from .memory import _FIELDS, _TOKEN, _WAITER, _TOKENS_AT, _WAITERS_AT, TOKENS, WAITERS
        import numpy as np
        ledger = self.governor._ledger
        states = {1: "RUNNING", 2: "SUSPENDED", 3: "WAITING", 4: "QUEUED", 5: "RESIDENT"}
        labels = [bytes(value).decode("utf-8", "replace") for value in ledger.l["name"]]
        def records(dtype, count, offset):
            rows = np.ndarray((count,), dtype=dtype, buffer=ledger.map, offset=offset).copy()
            result = []
            for row in rows[rows["id"] != 0]:
                entry = {name: int(row[name]) for name in dtype.names}
                entry["label_index"] = entry["label"]
                entry["label"] = labels[entry["label"]]
                if "state" in entry:
                    entry["state_code"] = entry["state"]
                    entry["state"] = states.get(entry["state"], "unknown")
                result.append(entry)
            return result
        error = (f"Memory ledger stalled: waiters with no grants, starts or releases for {elapsed / 60:.2f} minutes; "
                 f"process tree used {cores:.3f} CPU cores. Failing the run; checkpoints are kept for resume.")
        report = dict(error=error, owner_pid=self.pid, ledger=str(ledger.path),
                      snapshot="lock-free diagnostic; concurrent fields may be sampled at different instants",
                      elapsed_seconds=elapsed, cpu_cores=cores, processes=cpu,
                      header={name: int(ledger.h[i]) for i, name in enumerate(_FIELDS)},
                      tokens=records(_TOKEN, TOKENS, _TOKENS_AT),
                      waiters=records(_WAITER, WAITERS, _WAITERS_AT))
        path = self.diagnostic_directory / f"memory-stall-{self.pid}-{time.time_ns()}.json"
        data = json.dumps(report, indent=2) + "\n"
        try:
            path.write_text(data)
        except OSError as exc:
            error += f" Diagnostic file unavailable: {exc}."
        # Raw stderr avoids Python stream locks held by another thread in a stalled owner.
        try:
            os.write(2, (error + f"\nMemory-stall diagnostic: {path}\n" + data).encode())
        except OSError:
            pass                             # a closed log pipe must not prevent shutdown
        targets = [row for key, row in processes.items() if key[0] != self.pid]
        try:
            os.write(2, ("Memory-stall shutdown of verified run descendants: "
                         + json.dumps([row["pid"] for row in targets]) + "\n").encode())
        except OSError:
            pass
        from .memory import _alive
        for signum in (signal.SIGTERM, signal.SIGKILL):
            for row in reversed(targets):
                if _alive(row["pid"], row["started"]):
                    try:
                        os.kill(row["pid"], signum)
                    except ProcessLookupError:
                        pass
            if signum == signal.SIGTERM:
                time.sleep(1.0)
        os._exit(1)
