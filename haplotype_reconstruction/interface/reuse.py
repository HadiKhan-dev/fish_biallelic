"""The explicit record-reuse override: accept a run folder's records of named stages whatever code wrote them.

A stage's records are bound to its code (store.identity): edit a stage's source, even a comment, and its records and
everything computed from them are no longer found. That is the safe default. This module is the owner's explicit way
round it for one run folder: ``--reuse-records STAGE ...`` names stages whose existing records are to be accepted
after an edit that the user judges not to change results (a refactor, a speed-up with identical output).

How it works. After every successful workflow command the run folder remembers, per stage, the package sources under
which that stage's records were last read or written consistently (``CHECKPOINTS/code/<digest>.zip`` and
``CHECKPOINTS/code/ledger.json``). For a reused stage the code identity of its record derivations is computed from
that remembered snapshot instead of the installed sources (store.identity.pin_stage_sources), in the coordinator and
in every worker, so the existing records keep their identities and everything downstream derives from them as before.
The compiled-kernel cache is always keyed by the installed code.

The override is a state of the run folder, not of one command: once set it applies to every later command on the
folder (a later reader must find the same records), until ``--reuse-records none``. Every change of the state and
every computing command run under it is written to the ledger (reading commands write nothing: READ_ONLY), and
``haplotypes status`` shows it. A run made with it is reproducible only as far as the user's judgement
that the edit did not matter holds: nothing here checks that.
"""

from __future__ import annotations

import fcntl
import json
import zipfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.errors import SettingsError
from ..core.files import atomic_path
from ..store.store import CODE_DIRECTORY

__all__ = ["stage_names", "requested", "install", "worker_install", "record", "state"]

DIRECTORY = CODE_DIRECTORY  # a layout entry the store accepts
LEDGER = "ledger.json"
NONE = "none"
ALL = "all"
# Commands that only read a run folder's records: they apply its reuse state but write nothing into the checkpoint
# folder (it may be archived or read-only), neither a ledger entry nor remembered sources.
READ_ONLY = frozenset({"status", "read", "report"})


def stage_names() -> tuple[str, ...]:
    """The stages that can be named: the packages under ``stages`` that own record derivations."""
    from ..stages.catalog import STAGES

    return tuple(sorted({name for name in map(_package, STAGES.values()) if name is not None}))


def _package(spec: Any) -> str | None:
    """The package under ``stages`` that holds a catalogue entry's tasks (None: not a stage package, e.g. reports)."""
    parts = spec.tasks.split(".")
    return parts[1] if len(parts) > 2 and parts[0] == "stages" else None


def stage_of(graph_stage: str) -> str | None:
    """The nameable stage (package) of a graph stage ("calibration_divisor" -> "painting")."""
    from ..stages.catalog import STAGES

    spec = STAGES.get(graph_stage)
    return None if spec is None else _package(spec)


def _directory(checkpoints: str | Path) -> Path:
    return Path(checkpoints) / DIRECTORY


@contextmanager
def _locked(checkpoints: str | Path) -> Iterator[Path]:
    """Hold the ledger's lock (scattered jobs of one run folder finish at the same time)."""
    directory = _directory(checkpoints)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (LEDGER + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield directory / LEDGER
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return dict(schema=1, stages={}, reused=[], history=[])
    ledger = json.loads(path.read_text())
    if ledger.get("schema") != 1:
        raise SettingsError(f"{path}: unsupported record-reuse ledger")
    return ledger


def _write(path: Path, ledger: dict[str, Any]) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n")


def state(checkpoints: str | Path) -> dict[str, Any]:
    """The run folder's ledger: per stage the remembered source digest, the reused stages and the history of uses."""
    return _read(_directory(checkpoints) / LEDGER)


def _save_snapshot(checkpoints: str | Path) -> str:
    """Save the installed package sources once per distinct content; returns their digest."""
    from ..store.identity import package_sources, source_digest

    digest = source_digest()
    path = _directory(checkpoints) / f"{digest}.zip"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(path) as temporary:
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, data in package_sources().items():
                    archive.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data)
    return digest


def _load_snapshot(checkpoints: str | Path, digest: str) -> dict[str, bytes]:
    path = _directory(checkpoints) / f"{digest}.zip"
    if not path.is_file():
        raise SettingsError(f"{path}: the remembered sources of a reused stage are missing")
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def requested(value: Iterable[str]) -> tuple[str, ...] | None:
    """A command's ``reuse_records`` setting: None (nothing asked: keep the run folder's state), an empty tuple
    (``none``: stop reusing), ``("all",)`` (every stage the run folder remembers sources for) or the stages to reuse
    from now on."""
    names = tuple(dict.fromkeys(value))
    if not names:
        return None
    if names == (NONE,):
        return ()
    if names == (ALL,):
        return names
    unknown = sorted(set(names) - set(stage_names()))
    if unknown:
        raise SettingsError(
            f"reuse_records: {', '.join(unknown)}: name stages from {', '.join(stage_names())}, or give all or none alone"
        )
    return tuple(sorted(names))


def _pin(checkpoints: str | Path, stages: Iterable[str], remembered: dict[str, str]) -> dict[str, str]:
    from ..store.identity import pin_stage_sources

    stages = tuple(stages)
    missing = [stage for stage in stages if stage not in remembered]
    if missing:
        raise SettingsError(
            f"reuse_records: this run folder remembers no sources for {', '.join(missing)}: a stage can be reused "
            "only after a command of this version has read or written its records without the override"
        )
    snapshots = {digest: _load_snapshot(checkpoints, digest) for digest in {remembered[stage] for stage in stages}}
    pin_stage_sources({stage: snapshots[remembered[stage]] for stage in stages})
    return {stage: remembered[stage] for stage in stages}


def install(checkpoints: str | Path, asked: tuple[str, ...] | None, command: str) -> dict[str, str]:
    """Apply the run folder's reuse state in this process, after changing it to ``asked`` when a command asks.

    Returns {stage: remembered source digest} of the stages now reused (empty: the default, every identity from the
    installed sources). A change of state and every command run under a non-empty state are written to the ledger."""
    from ..store.identity import pin_stage_sources, source_digest

    directory = _directory(checkpoints)
    if asked is None and not (directory / LEDGER).is_file():
        pin_stage_sources({})
        return {}
    if asked is None:
        current = state(checkpoints)
        if command in READ_ONLY or not current["reused"]:
            # nothing asked and nothing to record (a reading command, or no stage reused): apply the state without
            # writing, so archived and read-only folders serve every reading command
            pinned = _pin(checkpoints, tuple(current["reused"]), current["stages"])
            if pinned and command != "status":  # status prints its own report (records_reused)
                _announce(pinned)
            return pinned
    with _locked(checkpoints) as path:
        ledger = _read(path)
        before = tuple(ledger["reused"])
        stages = before if asked is None else tuple(sorted(ledger["stages"])) if asked == (ALL,) else asked
        pinned = _pin(checkpoints, stages, ledger["stages"])
        if stages or stages != before:
            ledger["reused"] = list(stages)
            ledger["history"].append(
                dict(
                    time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    command=command,
                    reused=pinned,
                    installed=source_digest(),
                )
            )
            _write(path, ledger)
    if pinned:
        _announce(pinned)
    return pinned


def _announce(pinned: dict[str, str]) -> None:
    print(
        "[reuse] records of " + ", ".join(pinned) + " are accepted under the sources this run folder remembers, "
        "not the installed code (reuse_records; stop with --reuse-records none)",
        flush=True,
    )


def worker_install(checkpoints: str, stages: tuple[str, ...]) -> None:
    """Worker hook: pin the same stages as the coordinator (engine.runtime worker_hooks)."""
    if stages:
        _pin(checkpoints, stages, state(checkpoints)["stages"])


def record(checkpoints: str | Path, verified: Iterable[str], reused: Iterable[str]) -> None:
    """After a successful workflow command: remember the installed sources for every stage whose records the
    command read or wrote consistently (``verified``, graph stages or stage names), except the reused ones, whose
    remembered sources stay those of the code that wrote their records."""
    names = set()
    for stage in verified:
        name = stage if stage in stage_names() else stage_of(stage)
        if name is not None:
            names.add(name)
    names -= set(reused)
    if not names:
        return
    digest = _save_snapshot(checkpoints)
    with _locked(checkpoints) as path:
        ledger = _read(path)
        if all(ledger["stages"].get(name) == digest for name in names):
            return
        ledger["stages"].update(dict.fromkeys(sorted(names), digest))
        _write(path, ledger)
