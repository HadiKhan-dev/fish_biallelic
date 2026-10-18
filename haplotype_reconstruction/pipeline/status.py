"""`haplotypes status RUN`: run records and checkpoint progress, without decoding scientific payloads.

Run records (OUTPUT/run_records) give each invocation's last persisted state; the checkpoint store gives
genome-wide stage completion (run.ckpt complete/<stage>), and per contig each stage's own progress summary
(`<stage>.records.progress(file)`, which reads record keys and small plans only) plus live work states.
Problems (another layout, an unreadable container) are listed under "errors" instead of raised.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re

from ..store.registry import STAGES

_CONTIG_STAGES = tuple(stage.name for stage in STAGES if stage.scope == "contig")


def _natural_key(name: str):
    """chr2 before chr10."""
    return [int(part) if index % 2 else part for index, part in enumerate(re.split(r"(\d+)", name))]


def _records(directory: Path) -> list:
    rows = []
    for path in sorted((directory / "run_records").glob("*.json")):
        data = json.loads(path.read_text())
        stages = data.get("stages", [])
        rows.append(dict(record=str(path), command=data.get("command"), state=data.get("state"),
                         host=data.get("host"), pid=data.get("pid"), wall_seconds=data.get("wall_seconds"),
                         updated_utc=data.get("updated_utc"), active_stages=data.get("active_stages", []),
                         error=data.get("error"),
                         exclusive_stage_seconds={
                             name: sum(row["exclusive_seconds"] for row in stages if row["stage"] == name)
                             for name in sorted({row["stage"] for row in stages})}))
    return rows


def _progress(stage: str, file):
    return importlib.import_module(f"haplotype_reconstruction.{stage}.records").progress(file)


def _work_states(directory: Path) -> dict:
    from ..store.codec import PAYLOAD_SUFFIX
    states = {}
    try:
        contigs = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    except FileNotFoundError:
        return states
    for contig in contigs:
        rows = []
        try:
            entries = sorted(os.scandir(directory / contig), key=lambda entry: entry.name)
        except FileNotFoundError:
            continue
        for entry in entries:
            if entry.name.endswith(PAYLOAD_SUFFIX):
                try:
                    stat = entry.stat()
                except FileNotFoundError:
                    continue
                rows.append(dict(name=entry.name[:-len(PAYLOAD_SUFFIX)], size_bytes=stat.st_size,
                                 modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()))
        if rows:
            states[contig] = rows
    return states


def run_status(output, *, checkpoints=None, contigs=None) -> dict:
    """The status report of a run directory (see the module docstring)."""
    from ..store.store import CheckpointStore, WORK_DIRECTORY
    from .driver import RunPaths
    directory = Path(output).expanduser().resolve()
    root = (RunPaths.from_directory(directory).checkpoints if checkpoints is None
            else Path(checkpoints).expanduser().resolve())
    report = dict(output=str(directory), records=_records(directory), checkpoints=str(root), run=None,
                  contigs={}, errors=[])
    if not root.is_dir():
        if not directory.is_dir():
            report["errors"].append(f"{directory}: no such run directory")
        return report
    try:
        store = CheckpointStore(root)
    except RuntimeError as error:
        report["errors"].append(str(error))
        return report
    try:
        run = store.run
        manifest = run.get("manifest/run") if run.has("manifest/run") else {}
        report["run"] = dict(complete=list(run.keys("complete/")), identities=list(run.keys("identity/")),
                             contigs=manifest.get("contigs"))
    except (OSError, ValueError) as error:
        report["errors"].append(str(error))
        manifest = {}
    work = _work_states(root / WORK_DIRECTORY)
    names = list(contigs) if contigs else list(dict.fromkeys(
        [*(manifest.get("contigs") or ()), *sorted({*store.contig_names(), *work}, key=_natural_key)]))
    for name in names:
        try:
            file = store.contig(name)
            row = dict(size_bytes=file.size(), stages={})
            if file.exists():
                for stage in _CONTIG_STAGES:
                    row["stages"][stage] = _progress(stage, file)
        except (OSError, ValueError, KeyError) as error:
            report["errors"].append(f"{name}: {error}")
            row = dict(error=str(error))
        row["work"] = work.get(name, [])
        report["contigs"][name] = row
    return report


def _describe(stage: str, progress) -> str:
    """One stage's progress as short text: done, or its small fields (stage-defined)."""
    if progress is None:
        return "-"
    if isinstance(progress, dict):
        if progress.get("complete") is True or progress.get("done") is True or progress.get(stage) is True:
            return "done"
        parts = []
        for key, value in progress.items():
            if key in ("stage", "complete", "done", stage, "decoded_bytes") or value is None or value is False:
                continue
            if isinstance(value, (list, tuple)):
                # short lists of names or numbers (e.g. local selection's completed rounds, assembly levels)
                if value and len(value) <= 12 and all(isinstance(item, (str, int)) for item in value):
                    parts.append(f"{key} {','.join(map(str, value))}")
            elif not isinstance(value, dict):
                parts.append(f"{key} {value}")
        return " ".join(parts) or "-"
    return str(progress)


def format_status(report: dict) -> str:
    """The status report as text: run records, genome-wide stages, one line per contig."""
    lines = [f"Run records: {len(report['records'])} in {report['output']}/run_records"]
    for row in report["records"]:
        lines.append(f"  {Path(row['record']).stem}  {row['command']} {row['state']}  {row['host']} pid "
                     f"{row['pid']}  wall {row['wall_seconds'] or 0:.0f} s  updated {row['updated_utc']}")
        if row["error"]:
            lines.append(f"    error: {row['error']}")
        if row["active_stages"]:
            lines.append("    active: " + " > ".join(f"{stage['stage']} {stage.get('contig') or ''}".strip()
                                                     for stage in row["active_stages"]))
    lines.append(f"Checkpoints: {report['checkpoints']}")
    if report["run"] is not None:
        lines.append("  complete: " + (" ".join(name[len("complete/"):] for name in report["run"]["complete"])
                                       or "-"))
    for name, row in report["contigs"].items():
        if "error" in row:
            lines.append(f"  {name}: unreadable ({row['error']})")
            continue
        text = " | ".join(f"{stage} {_describe(stage, progress)}" for stage, progress in row["stages"].items())
        if row["work"]:
            text += " | work: " + ", ".join(state["name"] for state in row["work"])
        lines.append(f"  {name}: {text or 'not started'}")
    lines.extend(f"ERROR: {message}" for message in report["errors"])
    return "\n".join(lines)
